### QUESTION 2 ###

from amnesiac import blurry_memory
import random
import bisect
import time

answer1 = "9_FR33D0M8"
answer2 = "M1N_I_M4X_"

''' 2 SECTIONS:
	SECTION 1: 
	Code adapted and implemented from the search.py file in AI: a modern approach
	Changes:
		- defined a new fitness function
		- defined gene_pool variable
		- modified recombine uniform to return strings
		- created a while loop at the end to test the performance of the algorithm
	SECTION 2:
	Code created by me followint the genetic algorithm pseudocode:
		- This code didn't work well since it would give false results,
		- There must be some programming bug since sometimes the fitness function value is stored as 1.0
		  but when it is checked again it doesn't yield the same result.
'''		

def fitness_function(password):
  # defins with only one argument so that it can be used within other max() and map function
  # even though blurry_memory can take multiple passwords at once, sometimes it doesn't return the same number of passwords
  # because of this I made it so that it only takes one password and returns a float value as its fitness since this it how the
  # fitness function is used in the rest of the program
  fitness = list(blurry_memory([password], 170219976, 0).values())
  return fitness[0]

def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w) # creates cumulative weights eg [w1, w1+w2, w1+w2+w3, etc.]
    # chooses a random number within the maximum value of the cumulative weight (random.uniform)
    # compares that number to the entries in totals and checks the index at which it is larger (bisect)
    # the resulting index returns the corresponding value in seq
    # that would be the random choice based on the weight of each item 
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))] 


def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
    """[Figure 4.8]"""
    for i in range(ngen):
        print("generation: ", i)

        # for each individual in the population
        # select 2 individuals based on their fitness
        # recombine their information
        # mutate one of their characters basd on the mutation probability
        population = [mutate(recombine_uniform(*select(2, population, fitness_fn)), gene_pool, pmut) for i in range(len(population))]


        # check if the required fitness is achieved, if so return the fittest individual

        fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
        if fittest_individual:
            return fittest_individual

    # if fittest individual was not found return the best result so far 
    print(max([fitness_fn(p) for p in population]))
    return max(population, key=fitness_fn)


def fitness_threshold(fitness_fn, f_thres, population):
    if not f_thres:
        return None

    # check the maximum fitness string
    fittest_individual = max(population, key=fitness_fn)
    # run the fitness function and check if its fitness is above the threshold
    if fitness_fn(fittest_individual) >= f_thres:
        return fittest_individual

    return None


def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
        population.append(''.join(new_individual))

    return population


def select(r, population, fitness_fn):
    # select two of the parents based on their probability of being chosen
    fitnesses = map(fitness_fn, population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(r)]

def recombine_uniform(x, y): 
  # picks 5 random characters from the first string
  # picks 5 random characters from the second string 
  # then the first half of the new string contains the characters from x 
  # the second half contains the characters from y
  # this increases the randomness
    n = len(x)
    result = [0] * n
    indexes = random.sample(range(n), n)
    for i in range(n):
        ix = indexes[i]
        result[ix] = x[ix] if i < n / 2 else y[ix]

    return ''.join(str(r) for r in result)


def mutate(x, gene_pool, pmut):

    # if the random value is less than the mutation probability
    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n) # get random integer between 0 and length of x
    r = random.randrange(0, g) # get random integer between 0 and amount of genes/variations

    new_gene = gene_pool[r]

    # insert new gene at random point in string
    return x[:c] + new_gene + x[c + 1:]

if __name__ == "__main__":
	initial_population = init_population(250, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", 10)
	# print(initial_population)

	runs = 60
	successes = 0
	times = []

	answer = ""

	for run in range(runs):
	  print("Run: ", run)
	  time1 = time.time()
	  answer = genetic_algorithm(initial_population, fitness_function, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", 0.99, 150, 0.1)
	  time2 = time.time()
	  print(answer)
	  if answer in ["9_FR33D0M8", "M1N_M4X_"]:
		successes += 1
	  
	  times.append(time2-time1)

	print("Runs: ", runs)
	print("Successes: ", successes)
	print("Times minutes: ", times)

''' SECTION 2 uncomment to test

def string_generator(pop_number):

  # # returns a list of n legal strings

  # letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  # numbers = "0123456789"

  # strings = []

  # for i in range(n):

  #   new_string = ""

  #   for j in range(10):
  #     random_parameter = random.randrange(1,4) # Choose of the type of parameter to add with equal probability

  #     if random_parameter == 1:
  #       letter = random.choice(letters) # chooses any letter with equal chances
  #       new_string += letter
  #     elif random_parameter == 2:
  #       number = random.choice(numbers)
  #       new_string += number
  #     elif random_parameter == 3:
  #       new_string += "_"

  #   strings.append(new_string)

  # return strings

  gene_pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
  g = len(gene_pool)
  population = []
  for i in range(pop_number):
      new_individual = [gene_pool[random.randrange(0, g)] for j in range(10)]
      population.append(''.join(new_individual))

  return population

def genetic_reproduce(x, y):
  # combines two strings at random points\
  # At least one character of the string must be changed
  # position = random.randrange(1,10) # randrange does not include 10
  
  # # reattach split strings before or after based on 50% chance
  # order = random.random()

  # if order <= 0.5:
  #   child = x[position:]+y[:position]
  # else:
  #   child = y[position:]+x[:position]

  # return child
  n = len(x)
  result = [0] * n
  indexes = random.sample(range(n), n)
  for i in range(n):
      ix = indexes[i]
      result[ix] = x[ix] if i < n / 2 else y[ix]

  return ''.join(str(r) for r in result)

def genetic_mutation(x):

  # letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  # numbers = "0123456789"

  # random_position = random.randrange(0,9) # Choose any of the 10 position with equal probability
  # random_parameter = random.randrange(1,4) # Choose of the type of parameter to add with equal probability

  # new_child = list(child) # convert string to list so that it can be modified

  # if random_parameter == 1:
  #   letter = random.choice(letters) # chooses any letter with equal chances
  #   new_child[random_position] = letter
  # elif random_parameter == 2:
  #   number = random.choice(numbers)
  #   new_child[random_position] = number
  # elif random_parameter == 3:
  #   new_child[random_position] = "_"

  # new_child = ''.join(new_child)

  # return new_child

  gene_pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
  n = len(x)
  g = len(gene_pool)
  c = random.randrange(0, n)  # take one random value between 0 and the length of the password string
  r = random.randrange(0, g)  # take one random value between 0 and the length of the gene pool

  new_gene = gene_pool[r] # choose random allowed character

  return ''.join(x[:c] + new_gene + x[c + 1:])

def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

def genetic_algorithm(population, fitness_function, reproduction_function, mutation_function):
  # population = a set of individuals
  # fitness_fn = a function that measures fitness of an individual

  mutation_probability = 0.15
  generations = 0

  false_positives = 0

  while generations < 10000:
    
    new_population = [] # used list because it needs to be ordered when calculating the corresponding probabilities    

    for i in range(len(population)):
      
      # get the probability for each element in the list using the fitness function
      fitnesses = []
      
      for pop in population:
        fitness = list(blurry_memory([pop], 170219976, 0).values())[0]
        fitnesses.append(fitness)

      sampler = weighted_sampler(population, fitnesses) # each element in the population has a certain probability to be chosen for reproduction based on the total_probability

      x,y = [sampler() for i in range(2)]

      child = reproduction_function(x,y)

      if random.random() <= mutation_probability: 
        child = mutation_function(child)

      new_population.append(child)
    
    population = new_population[:] # copy list by value
    generations += 1
    print(max(fitnesses))

    if max(fitnesses) >= 0.99:
      if blurry_memory([population[fitnesses.index(1.0)]], 170219976, 0).values() == 1.0 or false_positives == 10:
        print("Found the result")
        print(population)
        print(fitnesses)
        print(population[fitnesses.index(1.0)])
        print(generations)
        false_positives += 1
        return [population, fitnesses]
      else:
        pass

  return [population, fitnesses]

strings = string_generator(250)

result = genetic_algorithm(strings, blurry_memory, genetic_reproduce, genetic_mutation)
print (result)

'''