import math
import random
import re
import sys
class C_45:

	#Decision tree using C4.5 algorithm
	def __init__(self, datafile,namesfile):
		 #used to define the data objects we are going to use through the whole program
		 #contains the path to the data comma seperated file
		self.DataFilePath = datafile
		#contains the path to the names comma seperated file
		self.NamesFilePath = namesfile
		#list to store the data rows which themselves are inside list as well
		self.data = []
		#list to store the rows used for training data set
		self.traindata = []
		#list to store the rows used to validation
		self.testdata = []
		#list to store the class label names
		self.classes = []
		#variable to hold the number of attributes in the data set
		self.numofAttributes = -1
		#dict to hold the attributes as keys and their possible corresponding values in a list as values
		self.attribue_values = {}
		# stores the list of names of attributes
		self.attribute_names = []
		# holds the tree nodes created using nodestruture class
		self.Dtree = None
		#holds the rules extracted from the decision tree
		self.rules = []
		#holds the indent level of each rule in the d tree
		self.level = 0
		#holds the rules according to the indent level and order of occurance in d tree
		self.rules_in_order = []
		# holds all the test rows that have been classified
		self.prediction = []
		#holds all the classified rows which are correctly predicted according to the actual label
		self.matches = []
		# holds the accuracy of a dtree on a validation test set
		self.accuracy = 0
		# holds the accuracies of all the d trees which performed classification in the loop
		self.alist = []

	def get_format_data(self):
		# fetch the data set and info about the data set from the files
		#and retrieve and store the data set in a particular format according to feature data types and class labels in the domain
		#open the file which contains the class labels info and feature-value info for reading purpose
		with open(self.NamesFilePath, "r") as file:
			#the first line is the info about the class label names seperated by commas in the domain
			class_labels = file.readline()
			#adding the class labels names to the classes list
			for x in class_labels.split(","):
				self.classes.append(x.strip())
			#parsing the attribute names and their value types in the data set
			for l in file:
				#splitting the line for attribute names and values and storing it in a list; used as a temp computation
				[attribute, values] = [x.strip() for x in l.split(":")] # temp computation
				#splitting the values for a attribute by "," delimiter and string them in a list; used as a temp computation
				values = [x.strip() for x in values.split(",")] # temp computation
				#dictionary string the attribute names as keys and their values in a list as values.
				self.attribue_values[attribute] = values # dict for attribute and type of values it has
		# holds the number of features in the data set
		self.numofAttributes = len(self.attribue_values.keys()) # no of attributes
		#stores the names of the attributes in a list by retrieving them from the keys of the dictionary
		self.attribute_names = list(self.attribue_values.keys()) # list of attributes
		#open the data set file and store the rows in a list based on the type of data types of the features in the domain
		with open(self.DataFilePath, "r") as file:
			# iterate through every row in the file
			for l in file:
				#split all the elements in the row based on "," delimiter and append them to the list row
				#each row represented by a list named row
				#row = [x.strip() for x in l.split(",")]
				row = []
				for x in l.split(","):
					row.append(x.strip())
				# if it is a dicrete attribute data set i.e. the housevotes data set,
				# append the class label to the last instead of the front of the list
				if self.Is_Attribute_Discrete(self.attribute_names[0]):
					labl = row[0]
					#print(labl)
					row.remove(labl)
					row.append(labl)
				#print(row)
				# if the row is not empty then append the row to the data set main list
				if row != [] or row != [""]: #if row is not a empty list or list with "", then append
					self.data.append(row)
				#if the data set is housevotes then remove the missing value rows
				if self.Is_Attribute_Discrete(self.attribute_names[0]):
					if row != [] or row != [""]:
						for element in row:
							if element == "?":
								self.data.remove(row)
								break
							else:
								continue
			# shuffle the rows in the data set before splitting into 80-20 or doing 3fold cross validation
			random.shuffle(self.data)
			try:
				if str(sys.argv[1]) == "80-20": #splitting into 80-20 train-test ratio
					self.traindata = self.data[ : int(len(self.data)*0.8)]
					self.testdata = self.data[int(len(self.data)*0.8) : ]

				elif str(sys.argv[1]) == "3fold": #applying the 3 fold cross validation
					#print()
					length = len(self.data)
					#print(length)
					flength = length/3
					#print(flength)
					fold1 = self.data[ : int(flength)]
					test1 = self.data[ : int(flength)]
					#print("fold1")
					#print(fold1)
					fold2 = self.data[ int(flength) : int(flength + flength)]
					test2 = self.data[ int(flength) : int(flength + flength)]
					#print("fold2")
					#print(fold2)
					fold3 = self.data[int(flength + flength) : ]
					test3 = self.data[int(flength + flength) : ]
					#print("fold3")
					#print(fold3)
					fold12 = fold1 + fold2
					#print("fold12")
					#print(fold12)
					fold13 = fold1 + fold3
					#print("fold13")
					#print(fold13)
					fold23 = fold2 + fold3
					#print("fold23")
					#print(fold23)
					#alist = []
					ilist = [0,1,2]
					#print(ilist)

					for i in ilist:
						if i == 0:
							#print("1")
							print("1st round")
							self.traindata = fold12
							#print(self.traindata)
							self.testdata = test3
							#print(self.testdata)
							self.preformatdata()
							self.Generate_Tree()
							self.Print_Tree()
							self.test_data()
							#print("accuracy : ", self.accuracy)
							self.alist.append(self.accuracy)
							self.postformatdata()
							continue
						elif i == 1:
							#print("2")
							print("2nd round")
							#print("fold1")
							#print(fold1)
							#print("fold2")
							#print(fold2)
							self.traindata = fold13
							#print(self.traindata)
							self.testdata = test2
							#print(self.testdata)
							self.preformatdata()
							self.Generate_Tree()
							self.Print_Tree()
							#print("here")
							self.test_data()
							#print("accuracy : ", self.accuracy)
							self.alist.append(self.accuracy)
							self.postformatdata()
							continue
						elif i == 2:
							#print("3")
							print("3rd round")
							self.traindata = fold23
							self.testdata = test1
							self.preformatdata()
							self.Generate_Tree()
							self.Print_Tree()
							self.test_data()
							#print("accuracy : ", self.accuracy)
							self.alist.append(self.accuracy)
							break
					avgaccuracy = 0
					for x in self.alist:
						#print("yes")
						avgaccuracy = avgaccuracy + x
					print(self.alist)
					print("avg accuracy : ", avgaccuracy/3)

				else:
					print("enter a valid option")
					sys.exit()
			except:
				print("Please enter a proper format")
				sys.exit()

			#print(self.traindata)
			#print(self.testdata)
	#convert the elements in the rows of the data set, if the features in the data set are continuous
	def preformatdata(self):
		#iterate through the rows in traindata
		for index,row in enumerate(self.traindata):
			#ranging from the index of 1st attribute to the last attribute in the domain
			for attr_index in range(self.numofAttributes):
				#if the attribute at that index is not discrete then
				#convert all the elements of the corresponding row from strings to floats
				if(not self.Is_Attribute_Discrete(self.attribute_names[attr_index])):
					self.traindata[index][attr_index] = float(self.traindata[index][attr_index]) # if not discrete then make the attribute elements in row list to be float data type
					#self.testdata[index][attr_index] = float(self.testdata[index][attr_index])
		#print(self.traindata)
	# convert the elements in the rows of the data set after training the d tree,
	#if the features in the data set are continuous
	def postformatdata(self):
		#iterate through the rows in traindata
		for index,row in enumerate(self.traindata):
			#ranging from the index of 1st attribute to the last attribute in the domain
			for attr_index in range(self.numofAttributes):
				#if the attribute at that index is not discrete then
				#convert all the elements of the corresponding row from floats to strings
				if(not self.Is_Attribute_Discrete(self.attribute_names[attr_index])):
					self.traindata[index][attr_index] = str(self.traindata[index][attr_index])
	#print the nodes in the trees, each node represents a rule
	def Print_Tree(self):
		#print(type(self.tree))
		# calls the print node method to get all the nodes in the dtree
		self.Print_Node(self.Dtree) # print node method gets the node returned by genrate tree
		#print(self.rules)
	#prints all the nodes returned by the generate tree
	def Print_Node(self, node, lev = 0 , indent=""):
		#rules = []
		# if not a leaf node
		if not node.is_Leaf:
			# if a discrete valued attribute for the node
			if node.threshold is None:
				#print("discrete")
				#discrete
				# iterate over child nodes
				for index,child in enumerate(node.children):
					# if child is a leaf node
					if child.is_Leaf:
						print(indent + node.label + " = " + self.attribue_values[node.label][index] + " : " + child.label)
						self.rules.append(str(lev) + " " + node.label + " = " + self.attribue_values[node.label][index] + " : " + child.label)
					else: # explore and print the other child nodes
						print(indent + node.label + " = " + self.attribue_values[node.label][index] + " : ")
						self.rules.append(str(lev) + " " + node.label + " = " + self.attribue_values[node.label][index] + " : ")
						self.Print_Node(child, lev + 1, indent + "	")
				#print(self.rules)
			else:
				#numerical continuous valued attribute
				#first child is left child
				leftChild = node.children[0]
				#second child is right child
				rightChild = node.children[1]
				# if the left child is leaf
				if leftChild.is_Leaf:
					print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
					self.rules.append(str(lev) + " " + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
				else: #explore and print the child nodes
					#self.level = self.level + 1
					print(indent + node.label + " <= " + str(node.threshold)+" : ")
					self.rules.append(str(lev) + " " + node.label + " <= " + str(node.threshold)+" : ")
					self.Print_Node(leftChild, lev + 1, indent + "	")
				#print(self.rules)

				#if the right child is leaf
				if rightChild.is_Leaf:
					print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
					self.rules.append(str(lev) + " " + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
				else: #explore and print the child nodes
					#self.level = self.level - 1
					print(indent + node.label + " > " + str(node.threshold) + " : ")
					self.rules.append(str(lev) + " " + node.label + " > " + str(node.threshold) + " : ")
					self.Print_Node(rightChild , lev + 1, indent + "	")
				#print(self.rules)
		#print(self.rules)


	#print(rules)
	# generate the d tree using the training data set and the atribute names list
	def Generate_Tree(self): # gets nodes returned from recursiveGenerateTree method
		#print(self.traindata)
		# the d tree holds all the nodes in the d tree returned by the recursive Generate Tree method
		self.Dtree = self.Recursive_Tree(self.traindata, self.attribute_names) #passing these as the values for the parameter in the function
	# the d tree gets build from bottom to top recursively
	def Recursive_Tree(self, current_Data, current_Attributes): # the DT builds from bottom to top
		#print(type(curData))
		#print((len(curData)))
		#check if all the examples in the train set have same class label
		All_Same = self.All_Same_Class(current_Data)
		# all if blocks are the stopping condition in the process of building the DT
		 # if no examples then return node object with True, "Fail", None arguments
		if len(current_Data) == 0:
			#Fail
			return Node(True, "Fail", None)
		# if all left examples have the same class label
		elif All_Same is not False: # if all examples correspond to same class label
			#return a node with that class
			# return a node object with true, that class label, none arguments
			return Node(True, All_Same, None)
		# if no attributes left to work on, then return the majority class label in the remaining class labels
		elif len(current_Attributes) == 0: # if no atributes to work on
			#return a node with the majority class
			major_class_label = self.Majority_Class(current_Data)
			#return a node object with true, majority class label, none arguments
			return Node(True, major_class_label, None)
			# if none of the conditions above specify then work on the attribute
			#by calling the split attribute method and use the remaining train data set
		else:
			#returns the attribute with the best information gain,
			#threshold if continuous and the splited data set by that attribute
			(best_attribute,best_attribute_threshold,data_splitted) = self.splitAttribute(current_Data, current_Attributes)
			#compute the new list of remaining attributes to work on
			r_attributes = current_Attributes[:]
			#remove the last best attribute
			r_attributes.remove(best_attribute) # remove the attribute used as a node in the tree
			#return the node abject with false, attribute name, threshold arguments
			node = Node(False, best_attribute, best_attribute_threshold) # label here is the attribute we are using to build the node
			#call the recursive tree method to build a node object for all the child nodes of the node we are working on.
			node.children = [self.Recursive_Tree(data_subset, r_attributes) for data_subset in data_splitted] # recursively builds the tree
			# class node object
			return node #this is an object of the class node returned with arguments
	# find the majority class in the train examples
	def Majority_Class(self, current_Data):
		# list with elements equal to class labels in the domain, every element set to 0
		label_frequency = [0]*len(self.classes)
		#print(type(label_frequency))
		#print(label_frequency)
		# iterate through every row in the train data set
		for row in current_Data:
			#get the index of the class label in the row
			label_index = self.classes.index(row[-1])
			# increase the count of the class label that appeared in the row
			label_frequency[label_index] += 1
		#get the index of the class label with the max frequency
		label_maxIndex = label_frequency.index(max(label_frequency))
		#return the corresponding class label of the max index
		return self.classes[label_maxIndex] # return the class label which occured the most in training data set

	#check if all the samples have the same class label
	def All_Same_Class(self, w_data): # check if all examples correspond to one class label
		# if no rows in w_data set, then return false
		if len(w_data) == 0:
			return False
		# iterate over the rows in w_data set
		for row in w_data:
			# if there is a mismatch, return false
			if row[-1] != w_data[0][-1]:
				return False
		#print(curData)
		return w_data[0][-1] # if yes, return that class label
	# check for if the attribute in the data set is discrete or not
	def Is_Attribute_Discrete(self, attribute):
		#if attribute not present in the attribute list, then display error
		if attribute not in self.attribute_names:
			#raise error
			raise ValueError("Attribute not in the attribute list")
		# if the attribute has only exactly one value and that value is continuous, then it is not discrete
		elif len(self.attribue_values[attribute]) == 1 and self.attribue_values[attribute][0] == "continuous":
			return False
		# it is discrete
		else:
			return True
	#given train data set and attribute names list, return the best attribute
	#which has the highest information gain. return the atribute name, threshold if it is a continuous
	#and the splitted train data based on that attribute value names
	def splitAttribute(self, current_Data, current_Attributes):
		# stores the subset of train data based on the values attribute can take
		data_splitted = []
		#maximum info gain among all the info gains for the attributes
		max_Info_Gain = -1*float("inf")
		#attribute with the max info gain
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_attribute_threshold = None
		#iterate over all the attributes in the remaining attribute list
		for attribute in current_Attributes:
			#get the index of the attribute we are working on
			index_Of_Attribute = self.attribute_names.index(attribute)
			#if the attribute is discrete
			if self.Is_Attribute_Discrete(attribute): # for attributes having discrete values
				# we split current_Data into n-subsets, depending on the n differnt values that attribute can take.
				#Choose the attribute with the max info gain
				#get all the values for the attribute
				Values_Of_Attribute = self.attribue_values[attribute]
				#create n subsets for n values of a attribute
				data_subsets = [[] for a in Values_Of_Attribute] # subsets is a list which has lists, each corresponds to a value that attribute can take
				#iterate over the rows in the data set and partition them in differnt dubsets of data
				#based on the different values of the attribute we are working on
				for row in current_Data: # iterate over all rows in data set, go over all the rows and put them in subset of lists for all the possible values of the attribute
					#i = range(len(row))
					## iterate over all the values that attribute can take
					for index in range(len(Values_Of_Attribute)):
						# if that particular value of the attribute is in that row
						if row[index_Of_Attribute] == Values_Of_Attribute[index]:
							# add that row to the list in subsets which is for that particular value there already
							data_subsets[index].append(row)
							# break the loop if that particular row has that particular value of the attribute
							# or go to next itertation to check if that row has the next value of the attribute
							break
				# return the info gain of the attribute based on train data set and
				#the subset of samples belogning to each value of the attribute
				g = self.gain(current_Data, data_subsets)
				if g > max_Info_Gain:
					max_Info_Gain = g
					#subsets is a list which has lists, each of which has training examples corresponding to values that attribute can take
					data_splitted = data_subsets
					best_attribute = attribute
					best_attribute_threshold = None
			else: # for continuous attributes
				#first sort the values for a feature we are working on in the data set,
				#make adjacent pairs of values of that feature and compute the threshold by taking the mean,
				# take the threshold which results in best info gain for that attribute,
				# subsequently take the best attribute
				# sort the data set based on the element(in the lists corresponding to rows) corresponding to the attribute index
				#of the attribute we are working with.
				current_Data.sort(key = lambda x: x[index_Of_Attribute])
				# loop to check for each pair of different attribute values in the consecutive rows of the data set
				for j in range(0, len(current_Data) - 1):
					# if the consecutive rows in data set do not have same value for the same attribute
					if current_Data[j][index_Of_Attribute] != current_Data[j+1][index_Of_Attribute]:
						#compute the threshold
						threshold = (current_Data[j][index_Of_Attribute] + current_Data[j+1][index_Of_Attribute]) / 2
						# append the rows with attribute value less than the threshold in less list
						low = []
						# append the rows with attribute value greater than the threshold in greater list
						high = []
						#append accordingly loop
						for row in current_Data:
							if(row[index_Of_Attribute] > threshold):
								high.append(row)
							else:
								low.append(row)
						# return the info gain for the threshold we are working at for the numerical valued attribute.
						g = self.gain(current_Data, [low, high])
						if g >= max_Info_Gain:
							# training examples greater than and less than the threshold value
							data_splitted = [low, high]
							#update accordingly
							max_Info_Gain = g
							best_attribute = attribute
							# threshold suggests the numerical value at which that continuous valued attribute
							 #divides the training examples most lopsidedly
							best_attribute_threshold = threshold
		return (best_attribute,best_attribute_threshold,data_splitted)
	#compute the info gain for an attribute
	def gain(self,MainSet, data_subsets):
		#input : data and disjoint subsets of data based on the values that attribute can take
		#output : information gain for that attribute
		S = len(MainSet)
		#calculate impurity before split
		Impurity_B_Split = self.entropy(MainSet)
		#calculate impurity after split
		# assign weights to subsets of each value for an attribute i.e. outer proportion of the no of examples
		# for that value of the attribute divided by total examples in the data set
		data_subset_Weights = [len(data_subset)/S for data_subset in data_subsets]
		Impurity_A_Split = 0
		for i in range(len(data_subsets)):
			# sumission over all the entropy's of the subsets representing differnt values of the attribute
			Impurity_A_Split += data_subset_Weights[i]*self.entropy(data_subsets[i])
		#calculate total gain
		Total_Info_Gain = Impurity_B_Split - Impurity_A_Split
		return Total_Info_Gain
	#compute the entropy for a attribute-value pair given a corresponding data set representing the particular attribute-value pair
	def entropy(self, W_Data_Set):
		# no of rows in the working data set
		S = len(W_Data_Set)
		#if no rows in the working data set return 0
		if S == 0:
			return 0
		# list counting the number of times each class label has appeared in the data set
		number_of_class_labels = [0 for i in self.classes]
		#iterate through all the rows in the working data set
		for row in W_Data_Set:
			#if type(row[1]) == float(): # housevotes discrete values, label is the first element
			# return the index of the class label from the classes list of that particular row.
			Class_Label_Index = list(self.classes).index(row[-1])
			#else:
				#classIndex = list(self.classes).index(row[0])
				# increment the count
			number_of_class_labels[Class_Label_Index] += 1
			# convert the count into ratio of no. of examples for that class label divided by total no. of examples
		number_of_class_labels = [x/S for x in number_of_class_labels]
		ent = 0
		for num in number_of_class_labels:
			#entropy formula
			ent += num*self.log(num)
		# the entropy formula has a submission over negative ratios
		return ent*-1

	# log to the base 2 method
	def log(self, x):
		#if x is 0 log base 2 is also 0
		if x == 0:
			return 0
		# else apply the log base 2 method of the math lib in python
		else:
			return math.log(x,2)

	# method to apply rules formed by the decision tree on the validation set and calulcating the accuracy for each turn
	# retrieves the rules formed by the decision trees using regular expressions
	def test_data(self):
			 # discrete data set
		if self.Is_Attribute_Discrete(self.attribute_names[0]):
			for rule in self.rules:
				rule = rule.strip()
				#print(rule)
				level_rule = re.findall("(^[0-9])\s([a-z].+?)\s([<|>|=].*?)\s([a-z])\s:(.*)",rule)
				attribute_value = re.findall("^.+?\s([a-z].+?)\s",rule)
				operator = re.findall("([<|>|=].*?)\s",rule)
				value = re.findall("\s([a-z])\s",rule)
				class_label = re.findall(":\s([a-z].+?$)",rule)
				#print(attribute_value)

				#print(operator)
				# print(operator)
				#print(value)
				#print(class_label)
				self.rules_in_order.append(level_rule)
		else:#continuous problem
			for rule in self.rules:
				rule = rule.strip()
				#print(rule)
				level_rule = re.findall("(^[0-9])\s([s|p].+?\s.+?)\s([<|>].*?)\s([0-9].*?)\s:(.*)",rule)
				attribute_value = re.findall("^.+?\s([s|p].+?\s.+?)\s",rule)
				#print(attribute_value)
				operator = re.findall("([<|>|=].*?)\s",rule)
				#print(operator)
				value = re.findall("\s([0-9].*?)\s",rule)
				#print(value)
				class_label = re.findall(":\s(I.+?$)",rule)
				#print(class_label)

				self.rules_in_order.append(level_rule)

		#print(self.rules_in_order)


			#print(type(=))
		for row in self.testdata: # every test example which is a list
			for rule in self.rules_in_order: # every rule is a tuple in a list
				#for t_rule in rule: # access the whole rule
				#print(rule[])
				level = rule[0][0] # first is the indent level
				#print(level)
				#print(type(rule[index][0]))
				a_index = self.attribute_names.index(rule[0][1]) # the column we are going to work in
				#print(index)
				a_name = rule[0][1]
				row[a_index]
				#print(ch)
				value = rule[0][3] # value, we are going to compare the example value with
				#print(value)
				cl = rule[0][4] # class label in the rule, if present
				#print(cl)
				if self.Is_Attribute_Discrete(self.attribute_names[0]): #applying the rules on the discrete data type attributes
					op = "==" # operator in the rule

					if eval("row[a_index]" + op + "value") is True and cl != "" : #1 case: if the rule applies and class label is present
							#print(row[index] + op + value)
						print(row, ":" + cl.strip())
						self.prediction.append([row, ":" + cl.strip()])
							#print(row[-1])
						if cl.strip() == row[-1].strip():
							print("It's a match")
							self.matches.append(["It's a match"])
						break
						# elif eval(row[index] + op + value) is True and cl == "": #2 case: if the rule applies and the class label is not present, go to the next rule in the list
						# 	break
					elif eval("row[a_index]" + op + "value") is False and cl == "":#3 case: if the rule does not apply and does not have a class label, then go to the next rule in the list
						#which has the same indent level but also has a class label, that rule will surely apply
						#self.goto_discrete(row, level, value)
						for index, rule in enumerate(self.rules_in_order): # every rule is a tuple in a list
							#for t_rule in rule: # access the whole rule
							if rule[0][0] == level and rule[0][1] == a_name and rule[0][3] != value: # if these conditions satisfy, the test example will be true for this rule
								if rule[0][4] != "":
									print(row, ":" + rule[0][4].strip())
									self.prediction.append([row, ":" + rule[0][4].strip()])
								#print(row[-1])
									if rule[0][4].strip() == row[-1].strip():
										print("It's a match")
										self.matches.append(["It's a match"])
									break
								else:
									level = rule[0][0]
									a_name = rule[0][1]
									value = rule[0][3]
									l_index = index
									#print([row, level, a_name, value, l_index])
									self.apply_recursive_rule(row, level, a_name, value, l_index)
									break
							else:
								continue
						break


					else:
						continue
				else: # appplying rules for the continuous data type attributes
					op = rule[0][2]
					#print(op)

					if eval(row[a_index] + op + value) is True and cl != "" : #1 case: if the rule applies and class label is present
						#print(row[index] + op + value)
						print(row, ":" + cl.strip())
						self.prediction.append([row, ":" + cl.strip()])
						#print(row[-1])
						if cl.strip() == row[-1].strip():
							print("It's a match")
							self.matches.append(["It's a match"])
						break
							# elif eval(row[index] + op + value) is True and cl == "": #2 case: if the rule applies and the class label is not present, go to the next rule in the list
							# 	break
					elif eval(row[a_index] + op + value) is False and cl == "":#3 case: if the rule does not apply and does not have a class label, then go to the next rule in the list
					#which has the same indent level but also has a class label, that rule will surely apply
						self.goto(row, level)
						break
					else:
						continue
								# elif eval(row[index] + op + value) is False and cl != "":#4 case: if the rule does not apply and there is a class label,
								# #then go to the next rule in the list, that will surely apply.
								# 	break
								# print(row[index] + op + value)
								# print("false")
			continue

		m = len(self.matches)
		p = len(self.prediction)
		self.accuracy = m/p*100
		print("accuracy : ", m/p*100 )
						#print(cl.strip())
	def goto(self, row, level): # use to handle the case 3 for continuous data type attributes
		for rule in self.rules_in_order: # every rule is a tuple in a list
			#for t_rule in rule: # access the whole rule
			if rule[0][0] == level and rule[0][4] != "": # if these conditions satisfy, the test example will be true for this rule
				print(row, ":" + rule[0][4].strip())
				self.prediction.append([row, ":" + rule[0][4].strip()])
				#print(row[-1])
				if rule[0][4].strip() == row[-1].strip():
					print("It's a match")
					self.matches.append(["It's a match"])

	def apply_recursive_rule(self, row, level, a_name, value, l_index): #used to handle the case 3 for the discrete data type attributes
		for index, rule in enumerate(self.rules_in_order): # every rule is a tuple in a list
			#for t_rule in rule: # access the whole rule
			#print(rule[])
			if index <= l_index:
				continue
			# if (rule[0][0] != level or rule[0][1] != a_name or rule[0][3] == value) or ():
			# 	continue
			level = rule[0][0] # first is the indent level
			#print(level)
			#print(type(rule[index][0]))
			a_index = self.attribute_names.index(rule[0][1]) # the column we are going to work in
			#print(index)
			row[a_index]
			a_name = rule[0][1]
			#print(ch)
			value = rule[0][3] # value, we are going to compare the example value with
			#print(value)
			cl = rule[0][4] # class label in the rule, if present
			#print(cl)

			op = "==" # operator in the rule

			if eval("row[a_index]" + op + "value") is True and cl != "" : #1 case: if the rule applies and class label is present
					#print(row[index] + op + value)
				print(row, ":" + cl.strip())
				self.prediction.append([row, ":" + cl.strip()])
					#print(row[-1])
				if cl.strip() == row[-1].strip():
					print("It's a match")
					self.matches.append(["It's a match"])
				break
				# elif eval(row[index] + op + value) is True and cl == "": #2 case: if the rule applies and the class label is not present, go to the next rule in the list
				# 	break
			elif eval("row[a_index]" + op + "value") is False and cl == "":#3 case: if the rule does not apply and does not have a class label, then go to the next rule in the list
				#which has the same indent level but also has a class label, that rule will surely apply
				#self.goto_discrete(row, level, value)
				for index, rule in enumerate(self.rules_in_order): # every rule is a tuple in a list
					#for t_rule in rule: # access the whole rule
					if rule[0][0] == level and rule[0][1] == a_name and rule[0][3] != value: # if these conditions satisfy, the test example will be true for this rule
						if rule[0][4] != "":
							print(row, ":" + rule[0][4].strip())
							self.prediction.append([row, ":" + rule[0][4].strip()])
						#print(row[-1])
							if rule[0][4].strip() == row[-1].strip():
								print("It's a match")
								self.matches.append(["It's a match"])
							break
						else:
							level = rule[0][0]
							a_name = rule[0][1]
							value = rule[0][3]
							l_index = index
							#print([row, level, a_name, value, l_index])
							self.apply_recursive_rule(row, level, a_name, value, l_index)
							break
					else:
						continue
				break
			else:
				continue

#node class to create node objects for d trees
class Node:
	#define the node structure in a d tree
	def __init__(self,is_Leaf, label, threshold):
		# carries the class label/attributename label for the node
		self.label = label
		#carries the threshold if it is a continuous data type attribute else None
		self.threshold = threshold
		#true if the node is a leaf node else false
		self.is_Leaf = is_Leaf
		# stores the child node of a node based on the values the attribute can take
		self.children = []
