#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_SPEED = 20 # m/s

def cartesian_distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

class WaypointUpdater(object):
    def __init__(self):

        rospy.init_node('waypoint_updater')
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.current_pose = None
        self.base_lane = None
        self.base_waypoints = None
        self.num_waypoints = 0
        self.base_waypoint_distances = None

        rospy.spin()

    # called when car's pose has changed
    # respond by emitting next set of final waypoints
    def pose_cb(self, msg):
        if self.base_waypoints == None:
            return

        rospy.loginfo("pose_cb::x:%f,y:%f,z:%f; qx:%f,qy:%f,qz:%f,qw:%f", 
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        self.current_pose = msg

        # find nearest waypoint
        wp1 = self.getNearestWaypointIndex(self.current_pose)
        self.nearestWaypointIndex = wp1
        rospy.loginfo("closest: %d", wp1)

        """
        # index of trajectory end
        wp2 = (wp1 + LOOKAHEAD_WPS)%self.num_waypoints

        # distance to end of trajectory
        d = self.distance(waypoints, wp1, wp2)

        # time to reach end
        t = d / MAX_SPEED

        p1 = self.base_waypoints[wp1].pose.pose.position
        p2 = self.base_waypoints[wp1].pose.pose.position
        jmt = JMT([], [], t)
        """

        # return next n waypoints as a Lane pbject
        waypoints = self.base_waypoints[wp1:(wp1 + LOOKAHEAD_WPS)%self.num_waypoints]
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)




        # TODO:

        #1. Calculate Frenet coordinates for current_pose

        #2. Calculate velocity in Frenet space

        #3. FSM to plan route

        #4. Compute final Frenet coordinate at time Tr (end of trajectory)

        #5. Fit polynomical jerk minimizing trajectory

        #6. Select points for spline, convert them to map coordinates

        #7. Generate splines for X and Y

        #7. Generate map coordinate points as fixed time intervals (t=0.2)

    def JMT(start, end, T):
        """
        Calculate the Jerk Minimizing Trajectory that connects the initial state
        to the final state in time T.

        INPUTS

        start - the vehicles start location given as a length three array corresponding to initial values of [s, s_dot, s_double_dot]
        end   - the desired end state for vehicle. Like "start" this is a length three array.
        T     - The duration, in seconds, over which this maneuver should occur.

        OUTPUT: an array of length 6, each value corresponding to a coefficent in the polynomial 
        s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5

        EXAMPLE:
        > JMT( [0, 10, 0], [10, 10, 0], 1)
        [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        """
        t2 = T * T
        t3 = t2 * T
        t4 = t2 * t2
        t5 = t3 * t2
        Tmat = np.array( [[t3, t4, t5], [3*t2, 4*t3, 5*t4], [6*T, 12*t2, 20*t3]] )

        Sf = end[0]
        Sf_d = end[1]
        Sf_dd = end[2]
        Si = start[0]
        Si_d = start[1]
        Si_dd = start[2]

        Sfmat = np.array( [[Sf - (Si + Si_d*T + 0.5*Si_dd*T*T)], [Sf_d - (Si_d + Si_dd*T)], [Sf_dd - Si_dd]] )
        alpha = np.linalg.inv(Tmat).dot(Sfmat)
        return (Si, Si_d, 0.5*Si_dd, alpha[0], alpha[1], alpha[2])

    # update nearest waypoint index by searching nearby values
    # waypoints are sorted, so search can be optimized
    def getNearestWaypointIndex(self, pose):  
        # func to calculate cartesian distance
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        # previous nearest point not known, do exhaustive search
        # todo: improve with binary search
        if self.nearestWaypointIndex == -1:    
            r = [(dl(wp.pose.pose.position, pose.pose.position), i) for i,wp in enumerate(self.base_waypoints)]
            return min(r, key=lambda x: x[0])[1]

        # previous nearest waypoint known, so scan points immediately after (& before)
        else:
            
            d = dl(self.base_waypoints[self.nearestWaypointIndex].pose.pose.position, pose.pose.position)
            # scan right
            i = self.nearestWaypointIndex
            d1 = d
            found = False
            while True:
                i = (i + 1) % self.num_waypoints
                d2 = dl(self.base_waypoints[i].pose.pose.position, pose.pose.position)
                if d2 > d1: break
                d1 = d2
                found = True
            if found:
                return i-1

            # scan left
            i = self.nearestWaypointIndex
            d1 = d
            found = False
            while True:
                i = (i - 1) % self.num_waypoints
                d2 = dl(self.base_waypoints[i].pose.pose.position, pose.pose.position)
                if d2 > d1: break
                d1 = d2
                found = True
            if found:
                return i+1

            return self.nearestWaypointIndex# keep prev value


    # Waypoint callback - data from /waypoint_loader
    # I expect this to be constant, so we cache it and dont handle beyond 1st call
    def waypoints_cb(self, base_lane):
        if self.base_lane == None:
            rospy.loginfo("waypoints_cb::%d", len(base_lane.waypoints))
            self.nearestWaypointIndex = -1
            self.base_lane = base_lane
            self.base_waypoints = base_lane.waypoints
            self.num_waypoints = len(self.base_waypoints)
            self.base_waypoint_distances = []
            d = 0.
            pos1 = self.base_waypoints[0].pose.pose.position
            for i in range(self.num_waypoints):
                pos2 = self.base_waypoints[i].pose.pose.position
                gap = cartesian_distance(pos1,pos2)
                self.base_waypoint_distances.append(d + gap)
                d += gap
                pos1 = pos2
            rospy.loginfo("track length: %f", d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    # get velocity of waypoint object
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    # set velocity at specified waypoint index
    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    # arguments: wapoints and two waypoint indices
    # returns distance between the two waypoints
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
