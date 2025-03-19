import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, learning_rate=1):
        """
        Initialize the neural network with the structure from the example.
        
        Parameters:
        learning_rate: The learning rate for weight updates (default=1)
        """
        # Initialize weights as shown in the diagram
        # Hidden layer weights (2 inputs to 2 hidden neurons)
        self.w3 = 0.1  # weight from input A to top hidden neuron
        self.w4 = 0.5  # weight from input B to top hidden neuron
        self.w5 = 0.3  # weight from input A to bottom hidden neuron
        self.w6 = 0.2  # weight from input B to bottom hidden neuron
        
        # Output layer weights (2 hidden neurons to 1 output neuron)
        self.w1 = 0.2  # weight from top hidden neuron to output
        self.w2 = 0.1  # weight from bottom hidden neuron to output
        
        # Learning rate
        self.learning_rate = learning_rate
        
        # For storing values during forward pass
        self.top_hidden_input = 0
        self.top_hidden_output = 0
        self.bottom_hidden_input = 0
        self.bottom_hidden_output = 0
        self.final_input = 0
        self.final_output = 0
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        
        Parameters:
        x: Input value
        
        Returns:
        float: Sigmoid of the input
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.
        
        Parameters:
        x: Input value (already passed through sigmoid)
        
        Returns:
        float: Derivative of sigmoid at the point
        """
        return x * (1 - x)
    
    def forward_pass(self, inputs):
        """
        Perform a forward pass through the network.
        
        Parameters:
        inputs: List or tuple containing input values [A, B]
        
        Returns:
        float: Network output
        """
        A, B = inputs
        
        # Calculate input and output for top hidden neuron
        self.top_hidden_input = (A * self.w3) + (B * self.w4)
        self.top_hidden_output = self.sigmoid(self.top_hidden_input)
        
        # Calculate input and output for bottom hidden neuron
        self.bottom_hidden_input = (A * self.w5) + (B * self.w6)
        self.bottom_hidden_output = self.sigmoid(self.bottom_hidden_input)
        
        # Calculate input and output for final neuron
        self.final_input = (self.top_hidden_output * self.w1) + (self.bottom_hidden_output * self.w2)
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output
    
    def backward_pass(self, inputs, target):
        """
        Perform a backward pass (backpropagation) to update weights.
        
        Parameters:
        inputs: List or tuple containing input values [A, B]
        target: Target output value
        
        Returns:
        float: Error value
        """
        A, B = inputs
        
        # Calculate output error
        output_error = target - self.final_output
        delta = output_error * self.sigmoid_derivative(self.final_output)
        
        # Update output layer weights
        w1_update = delta * self.top_hidden_output
        w2_update = delta * self.bottom_hidden_output
        
        # Calculate hidden layer errors
        delta1 = delta * self.w1 * self.sigmoid_derivative(self.top_hidden_output)
        delta2 = delta * self.w2 * self.sigmoid_derivative(self.bottom_hidden_output)
        
        # Update hidden layer weights
        w3_update = delta1 * A
        w4_update = delta1 * B
        w5_update = delta2 * A
        w6_update = delta2 * B
        
        # Apply weight updates
        self.w1 += self.learning_rate * w1_update
        self.w2 += self.learning_rate * w2_update
        self.w3 += self.learning_rate * w3_update
        self.w4 += self.learning_rate * w4_update
        self.w5 += self.learning_rate * w5_update
        self.w6 += self.learning_rate * w6_update
        
        return output_error
    
    def train(self, inputs, target, epochs=1):
        """
        Train the network for a specified number of epochs.
        
        Parameters:
        inputs: List or tuple containing input values [A, B]
        target: Target output value
        epochs: Number of training iterations
        
        Returns:
        list: Error history
        """
        error_history = []
        weight_history = []
        
        # Store initial weights
        weight_history.append({
            'w1': self.w1, 'w2': self.w2, 
            'w3': self.w3, 'w4': self.w4, 
            'w5': self.w5, 'w6': self.w6
        })
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward_pass(inputs)
            
            # Calculate and store error
            error = target - output
            error_history.append(error)
            
            # Backward pass
            self.backward_pass(inputs, target)
            
            # Store updated weights
            weight_history.append({
                'w1': self.w1, 'w2': self.w2, 
                'w3': self.w3, 'w4': self.w4, 
                'w5': self.w5, 'w6': self.w6
            })
            
            # Print progress
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}: Error = {error:.6f}, Output = {output:.6f}")
        
        return error_history, weight_history
    
    def print_network_state(self):
        """Print the current state of the network including all weights."""
        print("\nCurrent Network State:")
        print(f"Hidden Layer Weights:")
        print(f"  w3 (Input A to top hidden): {self.w3:.6f}")
        print(f"  w4 (Input B to top hidden): {self.w4:.6f}")
        print(f"  w5 (Input A to bottom hidden): {self.w5:.6f}")
        print(f"  w6 (Input B to bottom hidden): {self.w6:.6f}")
        print(f"Output Layer Weights:")
        print(f"  w1 (Top hidden to output): {self.w1:.6f}")
        print(f"  w2 (Bottom hidden to output): {self.w2:.6f}")

def verify_example():
    """
    Verify the neural network implementation against the example in the images.
    This replicates exactly what's shown in the images.
    """
    print("\n--- Verifying Example from Images ---")
    
    # Initialize network with learning rate = 1
    nn = SimpleNeuralNetwork(learning_rate=1)
    
    # Input values
    A, B = 0.1, 0.7
    
    # Expected target
    target = 1
    
    print(f"Inputs: A={A}, B={B}")
    print(f"Target: {target}")
    print(f"Initial weights:")
    print(f"  w1 (Top hidden to output): {nn.w1}")
    print(f"  w2 (Bottom hidden to output): {nn.w2}")
    print(f"  w3 (A to top hidden): {nn.w3}")
    print(f"  w4 (B to top hidden): {nn.w4}")
    print(f"  w5 (A to bottom hidden): {nn.w5}")
    print(f"  w6 (B to bottom hidden): {nn.w6}")
    
    # Forward pass
    output = nn.forward_pass([A, B])
    
    # Display forward pass results
    print("\nForward Pass Results:")
    print(f"Input to top hidden neuron = ({A}*{nn.w3})+({B}*{nn.w4})={nn.top_hidden_input:.6f}")
    print(f"Output of top hidden neuron (sigmoid) = {nn.top_hidden_output:.6f}")
    print(f"Input to bottom hidden neuron = ({A}*{nn.w5})+({B}*{nn.w6})={nn.bottom_hidden_input:.6f}")
    print(f"Output of bottom hidden neuron (sigmoid) = {nn.bottom_hidden_output:.6f}")
    print(f"Input to final neuron = ({nn.top_hidden_output}*{nn.w1})+({nn.bottom_hidden_output}*{nn.w2})={nn.final_input:.6f}")
    print(f"Final output (sigmoid) = {nn.final_output:.6f}")
    
    # Calculate output error
    error = target - output
    print(f"\nOutput error = Target - Output = {target} - {output:.6f} = {error:.6f}")
    delta = error * nn.sigmoid_derivative(output)
    print(f"Delta = Error * sigmoid_derivative(output) = {error:.6f} * {nn.sigmoid_derivative(output):.6f} = {delta:.6f}")
    
    # Update output layer weights (as shown in image 2)
    w1_update = delta * nn.top_hidden_output
    w2_update = delta * nn.bottom_hidden_output
    print(f"\nOutput Layer Weight Updates:")
    print(f"w1 update = delta * top_hidden_output = {delta:.6f} * {nn.top_hidden_output:.6f} = {w1_update:.6f}")
    print(f"w2 update = delta * bottom_hidden_output = {delta:.6f} * {nn.bottom_hidden_output:.6f} = {w2_update:.6f}")
    print(f"New w1 = {nn.w1} + {w1_update:.6f} = {nn.w1 + w1_update:.6f}")
    print(f"New w2 = {nn.w2} + {w2_update:.6f} = {nn.w2 + w2_update:.6f}")
    
    # Calculate hidden layer errors (as shown in image 2)
    delta1 = delta * nn.w1 * nn.sigmoid_derivative(nn.top_hidden_output)
    delta2 = delta * nn.w2 * nn.sigmoid_derivative(nn.bottom_hidden_output)
    print(f"\nHidden Layer Error Calculation:")
    print(f"delta1 = delta * w1 * sigmoid_derivative(top_hidden_output) = {delta:.6f} * {nn.w1} * {nn.sigmoid_derivative(nn.top_hidden_output):.6f} = {delta1:.6f}")
    print(f"delta2 = delta * w2 * sigmoid_derivative(bottom_hidden_output) = {delta:.6f} * {nn.w2} * {nn.sigmoid_derivative(nn.bottom_hidden_output):.6f} = {delta2:.6f}")
    
    # Update hidden layer weights (as shown in image 2)
    print(f"\nHidden Layer Weight Updates:")
    w3_update = delta1 * A
    w4_update = delta1 * B
    w5_update = delta2 * A
    w6_update = delta2 * B
    print(f"w3 update = delta1 * A = {delta1:.6f} * {A} = {w3_update:.6f}")
    print(f"w4 update = delta1 * B = {delta1:.6f} * {B} = {w4_update:.6f}")
    print(f"w5 update = delta2 * A = {delta2:.6f} * {A} = {w5_update:.6f}")
    print(f"w6 update = delta2 * B = {delta2:.6f} * {B} = {w6_update:.6f}")
    print(f"New w3 = {nn.w3} + {w3_update:.6f} = {nn.w3 + w3_update:.6f}")
    print(f"New w4 = {nn.w4} + {w4_update:.6f} = {nn.w4 + w4_update:.6f}")
    print(f"New w5 = {nn.w5} + {w5_update:.6f} = {nn.w5 + w5_update:.6f}")
    print(f"New w6 = {nn.w6} + {w6_update:.6f} = {nn.w6 + w6_update:.6f}")
    
    # Apply backpropagation
    nn.backward_pass([A, B], target)
    
    # Display updated weights
    print("\nUpdated weights after backward pass:")
    print(f"  w1 (Top hidden to output): {nn.w1:.6f}")
    print(f"  w2 (Bottom hidden to output): {nn.w2:.6f}")
    print(f"  w3 (A to top hidden): {nn.w3:.6f}")
    print(f"  w4 (B to top hidden): {nn.w4:.6f}")
    print(f"  w5 (A to bottom hidden): {nn.w5:.6f}")
    print(f"  w6 (B to bottom hidden): {nn.w6:.6f}")

def run_training_example():
    """
    Run a more complete training example to show how the network learns over time.
    """
    print("\n--- Running Training Example ---")
    
    # Initialize network
    nn = SimpleNeuralNetwork(learning_rate=0.5)
    
    # Input values and target
    A, B = 0.1, 0.7
    target = 1
    
    # Print initial state
    nn.print_network_state()
    
    # Train for 1000 epochs
    print("\nTraining for 1000 epochs...")
    error_history, weight_history = nn.train([A, B], target, epochs=1000)
    
    # Print final state
    nn.print_network_state()
    
    # Final forward pass
    final_output = nn.forward_pass([A, B])
    print(f"\nFinal output: {final_output:.6f}")
    print(f"Final error: {target - final_output:.6f}")
    
    # Plot error over time
    plt.figure(figsize=(10, 5))
    plt.plot(error_history)
    plt.title('Error over time')
    plt.xlabel('Epoch RANGES')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()
    
    # Plot weights over time
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.plot([w['w1'] for w in weight_history], label='w1')
    plt.plot([w['w2'] for w in weight_history], label='w2')
    plt.title('Output Layer Weights')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot([w['w3'] for w in weight_history], label='w3')
    plt.plot([w['w4'] for w in weight_history], label='w4')
    plt.title('Hidden Layer Weights (Top Neuron)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot([w['w5'] for w in weight_history], label='w5')
    plt.plot([w['w6'] for w in weight_history], label='w6')
    plt.title('Hidden Layer Weights (Bottom Neuron)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Verify the example from the images
    verify_example()
    
    # Run a more complete training example
    run_training_example()