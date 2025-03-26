import matplotlib.pyplot as plt    
from cmcrameri import cm

def visualize_neuron_activity(layer,activation_data, idx_n, row_length=3, output_file=None):

    cmap = cm.lajolla
    neuron_total_rows = (len(idx_n) + row_length - 1) // row_length 
    fig, axis_collection = plt.subplots(neuron_total_rows, row_length, figsize=(row_length * 2, neuron_total_rows * 2))
    axis_collection = axis_collection.flatten()
    
    for i, neuron_index in (idx_n):
        if neuron_index >= activation_data.shape[1]:  # Check bounds
            continue   
        current_axis = axis_collection[i]
        current_axis.imshow(activation_data[:, neuron_index].reshape(-1, 1), aspect='auto', cmap=cmap)
        current_axis.set_title(f'Neuron {neuron_index + 1}', fontsize=8)
        current_axis.tick_params(axis='both', which='major', labelsize=6)

    for remaining_axis in range(i + 1, len(axis_collection)):
        axis_collection[remaining_axis].axis('off')

    plt.tight_layout()
    plt.title(f'Layer : {layer}')
    
    
    
def visualize_neuron_activity_all(activation_data, display_count=50, row_length=10, output_file=None):
    """
    Displays activation patterns for a set of neurons.

    Args:
        activation_data (numpy.ndarray or torch.Tensor): 2D array of neuron activations.
        display_count (int): Number of neurons to visualize.
        row_length (int): Number of neurons to display per row.
        output_file (str, optional): Path to save the visualization. If None, displays the plot.
    """
    cmap = cm.lajolla
    
    neuron_total_rows = (display_count + row_length - 1) // row_length
    figure, axis_collection = plt.subplots(neuron_total_rows, row_length, figsize=(row_length * 2, neuron_total_rows * 2))
    axis_collection = axis_collection.flatten()

    for neuron_index in range(display_count):
        if neuron_index >= activation_data.shape[1]:
            break

        current_axis = axis_collection[neuron_index]
        current_axis.imshow(activation_data[:, neuron_index].reshape(-1, 1), aspect='auto', cmap=cmap)
        current_axis.set_title(f'Neuron {neuron_index + 1}', fontsize=8)
        current_axis.tick_params(axis='both', which='major', labelsize=6)

    for remaining_axis in range(neuron_index + 1, len(axis_collection)):
        axis_collection[remaining_axis].axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=600)
    else:
        plt.show()

def check_neuron(activation_data, decoded_activations, neuron_index):
    
    num_samples = activation_data.shape[0]

    if neuron_index >= activation_data.shape[1] or neuron_index >= decoded_activations.shape[1]:
        print(f"Error: Neuron index {neuron_index} is out of bounds.")
        return
    
    if activation_data.shape[1] == 1 :
        neuron_index = 1 
    
    original_neuron_data = activation_data[:, neuron_index]
    decoded_neuron_data = decoded_activations[:, neuron_index]

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_samples), original_neuron_data, label=f'Original Neuron {neuron_index}', alpha=0.7)
    plt.plot(range(num_samples), decoded_neuron_data, label=f'Reconstructed Neuron {neuron_index} (after SAE)', alpha=0.7)

    plt.xlabel('vector length')
    plt.ylabel('Activation Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_training_errors(reconstruction_losses, sparsity_penalties, total_losses):
    epochs = range(1, len(total_losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, reconstruction_losses, label='Reconstruction Loss')
    plt.plot(epochs, total_losses, label='Total Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction and Total Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, sparsity_penalties, label='Sparsity Penalty', color='orange')
    plt.plot(epochs, total_losses, label='Total Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Penalty/Loss')
    plt.title('Sparsity Penalty and Total Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()