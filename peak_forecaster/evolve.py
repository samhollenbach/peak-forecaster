import random
from network import Network
import tensorflow as tf
from tensorflow import keras
import evaluate

class NetworkEvolver:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_shape = len(x_test)
        self.output_shape = len(y_test)

        self.retain = 4
        self.random_select = 0.3
        self.mutate_chance = 0.4

        self.choices = {
            'layers': [1, 2, 3],
            'activation': ['elu', 'relu', keras.layers.LeakyReLU(alpha=0.2), 'tanh'],
            'nodes': [4, 8, 16, 32, 64],
            'loss_positive_scalar': [1.0, 2.0, 5.0, 7.5, 10.0, 15.0],
            'optimizer': [tf.keras.optimizers.RMSprop,
                          tf.keras.optimizers.Adamax,
                          tf.keras.optimizers.Nadam],
            'learning_rate': [0.01, 0.001, 0.0001]
        }


    def choose_random(self, param_name, network_layers=None):
        if param_name == 'nodes':
            if network_layers is None:
                raise ValueError("Must set layers parameter if picking random nodes")
            return [random.choice(self.choices['nodes']) for _ in range(network_layers)]
        return random.choice(self.choices[param_name])

    def get_random_params(self):
        network_layers = self.choose_random('layers')
        return {
            'layers': network_layers,
            'activation': self.choose_random('activation'),
            'nodes': self.choose_random('nodes', network_layers),
            'loss_positive_scalar': self.choose_random('loss_positive_scalar'),
            'optimizer': self.choose_random('optimizer'),
            'learning_rate': self.choose_random('learning_rate')
        }


    def create_population(self, count):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        """
        pop = []
        for _ in range(0, count):
            # Create a random network.
            params = self.get_random_params()
            network = Network(params)

            # Add the network to our population.
            pop.append(network)

        return pop

    def breed(self, mother, father):
        """

        :param Network mother:
        :param Network father:
        :return:
        """
        children = []
        for _ in range(2):

            child_params = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.choices:
                child_params[param] = random.choice(
                    [mother.params[param], father.params[param]]
                )

            # Now create a network object.
            network = Network(child_params)

            children.append(network)

        return children

    def mutate(self, network):
        """

        :param Network network:
        :return:
        """
        # Choose a random key.
        mutation = random.choice(list(self.choices.keys()))
        layers = None
        if mutation == 'nodes':
            layers = network.params['layers']
        # Mutate one of the params.
        network.params[mutation] = self.choose_random(mutation, network_layers=layers)

        return network

    def evolve(self, pop):
        """Evolve a population of networks.
        Args:
            pop (list): A list of network parameters
        """
        # Get scores for each network.
        graded = [(evaluate.evaluate_basic(network,
                                           self.x_train,
                                           self.y_train,
                                           self.x_test,
                                           self.y_test), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in
                  sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Randomly mutate some of the networks we're keeping.
        for individual in parents:
            if self.mutate_chance > random.random():
                individual = self.mutate(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents