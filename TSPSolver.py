#!/usr/bin/python3

from which_pyqt import PYQT_VER

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

		while len(visited) < len(cities) - 1:
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

	def branchAndBound(self, time_allowance=60.0):
		pass

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy(self, time_allowance=60.0):
		default_tour = self.defaultRandomTour(time_allowance)
		solution = default_tour['soln']
		route = solution.route
		cost = solution.cost
		solutions = 0

		start_time = time.time()
		improvement = True
		while improvement:
			improvement = False
			for i in range(1, len(route) - 2):
				for j in range(i + 1, len(route) - 1):
					print(i, j)
					if self.calculate_distance(i, i + 1, route) + self.calculate_distance(j, j + 1, route) > \
							self.calculate_distance(i, j, route) + self.calculate_distance(i + 1, j + 1, route):
						route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
						improvement = True
						solution = TSPSolution(route)
						route = solution.route
						cost = solution.cost
						solutions += 1

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
		cities = self._scenario.getCities
		city_one = route[city_one_index]
		city_two = route[city_two_index]

		return city_one.costTo(city_two)
