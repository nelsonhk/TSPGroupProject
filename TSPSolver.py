#!/usr/bin/python3

from which_pyqt import PYQT_VER
from TSPState import TSPState
from queue import PriorityQueue

if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
	def __init__(self, gui_view):
		self._scenario = None
		self.allowed_time = True
		self.unique_id = 0
		self.states = 0
		self.pruned = 0
		self.solutions = 0
		self.max_size = 0
		self.start_time = time.time()
		self.time_allowance = 60

	def setupWithScenario(self, scenario):
		self._scenario = scenario

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time() - start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation(ncities)
			route = []
			# Now build the route using the random permutation
			for i in range(ncities):
				route.append(cities[perm[i]])
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		start = cities[0]
		current = start
		total_distance = 0
		visited = [current]
		start_time = time.time()

		while len(visited) < len(cities):
			next_city = None
			min_distance = float('inf')
			for city in cities:
				if city not in visited:
					distance = current.costTo(city)
					if distance < min_distance:
						min_distance = distance
						next_city = city
			visited.append(next_city)
			current = next_city
			total_distance += min_distance

		total_distance += current.costTo(start)
		end_time = time.time()

		results['cost'] = total_distance
		results['time'] = end_time - start_time
		results['count'] = None
		results['soln'] = TSPSolution(visited)
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	# Time complexity: O(n!)
	# Space complexity: O(1)
	def branchAndBound(self, time_allowance=60.0):
		# initialize several variables that help us in the future
		cities = self._scenario.getCities()
		ncities = len(cities)
		visited = [0]
		found_tour = False
		self.set_variables()
		self.start_time = time.time()
		self.time_allowance = time_allowance

		# make cost matrix
		cost_matrix = self.make_cost_matrix()

		# reduce the matrix
		reduced_matrix, lower_bound = self.reduce_cost_matrix(cost_matrix)

		# get the initial bssf
		default_tour = self.defaultRandomTour(time_allowance)
		bssf = default_tour
		bssf_cost = bssf['cost']

		# make the queue
		queue = self.make_queue(reduced_matrix, lower_bound, ncities)

		# run while the length of the queue is grater than 0 and while time hasn't run out
		while len(queue) > 0 and self.allowed_time:
			# pop the route with the lowest bssf off the queue
			city = queue.pop(0)
			# if the lower bound is smaller than the bssf
			if city[2] < bssf_cost and self.allowed_time:
				# update the visited cities
				visited = city[5]
				# if the # of cities visited is = to the # of cities in the problem
				if len(visited) == ncities:
					# a tour is found
					found_tour = True
					# add each visited city to a route array
					route = []
					for i in range(len(visited)):
						route.append(cities[visited[i]])
					# give that route array to a TSPSolution object and that becomes new bssf, update the bssf cost
					bssf = TSPSolution(route)
					bssf_cost = bssf.cost
					# add to the number of solutions found
					self.solutions += 1
				# add the children states to the queue
				else:
					self.add_children(queue, city[2], city[3], city[4], city[5], ncities)
			# prune the states where the lower bound is greater than the bssf cost
			else:
				self.pruned += 1

		# create a dictionary with all the best route results
		end_time = time.time()
		results = {}
		results['cost'] = bssf_cost if found_tour else math.inf
		results['time'] = end_time - self.start_time
		results['count'] = self.solutions
		results['soln'] = bssf if found_tour else bssf['soln']
		results['max'] = self.max_size
		results['total'] = self.states
		results['pruned'] = self.pruned
		return results

	def make_cost_matrix(self):
		cities = self._scenario.getCities()
		ncities = len(cities)

		# initialize the cost matrix
		cost_matrix = [[0 for i in range(ncities)] for j in range(ncities)]

		# fill in the cost matrix with the lengths between cities
		for i in range(ncities):
			for j in range(ncities):
				city_one = cities[i]
				city_two = cities[j]
				cost_matrix[i][j] = city_one.costTo(city_two)

		return np.array(cost_matrix)

	def reduce_cost_matrix(self, cost_matrix):
		cities = self._scenario.getCities()
		ncities = len(cities)
		min_values = []
		lower_bound = 0

		# find the minimum values in each row
		for i in range(ncities):
			min_value = math.inf
			for j in range(ncities):
				if cost_matrix[i][j] < min_value:
					min_value = cost_matrix[i][j]
			if min_value < math.inf:
				min_values.append(min_value)
				lower_bound += min_value
			else:
				min_values.append(0)

		# subtract the minimum value of each row from each city in row
		for i in range(ncities):
			for j in range(ncities):
				cost_matrix[i][j] -= min_values[i]

		# clear the min values array
		min_values.clear()

		# find the minimum value in each col
		cost_matrix_transpose = cost_matrix.copy().transpose()
		for i in range(ncities):
			min_value = math.inf
			for j in range(ncities):
				if cost_matrix_transpose[i][j] < min_value:
					min_value = cost_matrix_transpose[i][j]
			if min_value < math.inf:
				min_values.append(min_value)
				lower_bound += min_value
			else:
				min_values.append(0)

		# subtract the minimum value of each col from each city in col
		for i in range(ncities):
			for j in range(ncities):
				cost_matrix[i][j] -= min_values[j]

		return cost_matrix, lower_bound

	def make_queue(self, cost_matrix, lower_bound, ncities):
		# make the queue
		queue = []
		heapq.heapify(queue)

		col = 1
		for i in range(1, ncities):
			child = cost_matrix.copy()
			child_cost = child[0][col]

			# the row and column values of the current cities
			for j in range(ncities):
				for k in range(ncities):
					if k == col:
						child[j][k] = math.inf
					if j == 0:
						child[j][k] = math.inf

			child[col][0] = math.inf

			# reduce the matrix
			reduced_child_matrix, child_lower_bound = self.reduce_cost_matrix(child)

			# create an array of previously visited cities
			cities = [0, i]
			level = 1

			# if the lower bound isn't infinity, push the state onto the queue
			if (lower_bound + child_lower_bound + child_cost) != math.inf:
				heapq.heappush(queue, (((lower_bound + child_lower_bound + child_cost) / level),
									   self.unique_id,
									   (lower_bound + child_lower_bound + child_cost),
									   level, reduced_child_matrix, cities))
				self.update_id()
				self.states += 1
				self.update_queue(queue)
			# if the lower bound is infinity, prune the state
			else:
				self.pruned += 1

			col += 1

			# check the time, if we are over the allowed time then return the current queue
			self.check_time()
			if not self.allowed_time:
				return queue

		return queue

	def add_children(self, queue, lower_bound, level, cost_matrix, visited, ncities):
		last_city = visited[-1]
		# finds the remaining cities left
		cities_remaining = self.cities_remaining(visited, ncities)
		index = 0
		col = cities_remaining[index]
		for i in range(0, ncities - len(visited)):
			child = cost_matrix.copy()
			child_cost = child[last_city][col]

			# the row and column values of the current cities
			for j in range(ncities):
				for k in range(ncities):
					if k == col:
						child[j][k] = math.inf
					if j == last_city:
						child[j][k] = math.inf

			child[col][last_city] = math.inf

			# reduce the matrix
			reduced_child_matrix, child_lower_bound = self.reduce_cost_matrix(child)

			# add current city to visited cities array
			cities = visited.copy()
			cities.append(col)
			# level identifier
			new_level = level + 1
			# if the lower bound isn't infinity, push the state onto the queue
			if (lower_bound + child_lower_bound + child_cost) != math.inf:
				heapq.heappush(queue, (((lower_bound + child_lower_bound + child_cost) / new_level),
									   self.unique_id,
									   (lower_bound + child_lower_bound + child_cost),
									   new_level, reduced_child_matrix, cities))
				self.update_id()
				self.states += 1
				self.update_queue(queue)
			# if the lower bound is infinity, prune the state
			else:
				self.pruned += 1

			index += 1
			# checks if the next index is out of bounds
			if index <= len(cities_remaining) - 1:
				col = cities_remaining[index]

			# check the time, if we are over the allowed time then return the current queue
			self.check_time()
			if not self.allowed_time:
				return queue
		return queue

	def cities_remaining(self, visited, ncities):
		# initialize an array6
		cities = []

		# append an index to cities for every city in the problem
		for i in range(ncities):
			cities.append(i)

		# remove the indexes of cities we have already visited
		for i in range(len(visited)):
			city = visited[i]
			cities.remove(city)

		# return the array of cities visited
		return cities

	def update_id(self):
		# increment the unique id variable
		self.unique_id += 1

	def update_queue(self, queue):
		# if the current length of the queue is greater than the max size, update the max size
		if len(queue) > self.max_size:
			self.max_size = len(queue)

	def check_time(self):
		# if the current time - the start is greater than the allowed, changed allowed time to false
		if (time.time() - self.start_time) > self.time_allowance:
			self.allowed_time = False

	def set_variables(self):
		self.unique_id = 0
		self.states = 0
		self.pruned = 0
		self.solutions = 0
		self.max_size = 0

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy(self, time_allowance=60.0):
		default_tour = self.greedy(time_allowance)
		solution = default_tour['soln']
		route = solution.route
		cost = solution.cost
		solutions = 0
		print(cost)

		start_time = time.time()
		improvement = True
		while improvement:
			improvement = False
			for i in range(1, len(route) - 2):
				for j in range(i + 1, len(route) - 1):
					if self.calculate_distance(i, i + 1, route) + self.calculate_distance(j, j + 1, route) > \
							self.calculate_distance(i, j, route) + self.calculate_distance(i + 1, j + 1, route):
						new_route = route.copy()
						new_route[i + 1:j + 1] = reversed(new_route[i + 1:j + 1])
						new_solution = TSPSolution(new_route)
						if new_solution.cost < solution.cost:
							improvement = True
							solution = TSPSolution(new_route)
							route = solution.route
							cost = solution.cost
							solutions += 1
		print(cost)

		end_time = time.time()
		results = {}
		results['cost'] = cost
		results['time'] = end_time - start_time
		results['count'] = solutions
		results['soln'] = solution
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	def calculate_distance(self, city_one_index, city_two_index, route):
		city_one = route[city_one_index]
		city_two = route[city_two_index]

		return city_one.costTo(city_two)
