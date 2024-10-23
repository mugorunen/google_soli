import os
import numpy as np
import h5py


class SoliDataImporter:
    """To load data from the datapath and convert it to images (3D numpy arrays)"""

    def __init__(self, data_directory=None):
        """
        :param data_directory (str): The path to the directory containing the .h5 files.
        """
        self.data_directory = data_directory

    def load_deep_soli_data(self):
        """
        Loads the Deep-Soli dataset from the specified directory.

        Returns:
        - stacked_radar_data (list of numpy arrays): The loaded radar data. Number of frames x 1024 x Number of channels
        - labels (list of numpy arrays): The corresponding labels for stacked frames
        """
        stacked_radar_data = []  # List to store radar data
        labels = []  # List to store labels

        # Iterate over all files in the data directory
        for file_name in os.listdir(self.data_directory):
            if file_name.endswith('.h5'):
                # Parse the label, participant, and session from the file name
                label_number, participant_number, session_number = file_name.replace('.h5', '').split('_')

                label_number = int(label_number)  # Convert label to integer
                
                # Load the .h5 file
                file_path = os.path.join(self.data_directory, file_name)
                with h5py.File(file_path, 'r') as f:
                    # Combine data from all channels
                    radar_data = []
                    for channel in range(4):
                        if f'ch{channel}' in f:
                            radar_data.append(f[f'ch{channel}'][()])  # Append the channel data

                    # Stack channel data along a new axis to create a multi-channel input
                    radar_data = np.stack(radar_data, axis=-1)  # Shape: (time_steps, 1024, 4)
                    
                    # Assign labels for each time step for the current movement
                    num_time_steps = radar_data.shape[0]
                    labels_for_this_file = np.full(num_time_steps, label_number) 

                    # Append the radar data and corresponding labels to the lists
                    stacked_radar_data.append(radar_data) 
                    labels.append(labels_for_this_file) 

        return stacked_radar_data, labels 
    
    def convert_to_images(self, stacked_radar_data, labels, target_shape=(32, 32)):
        """
        Concatenates all radar data and labels into numpy arrays.

        Parameters:
        - stacked_radar_data (list of numpy arrays): The radar data.
        - labels (list of numpy arrays): The corresponding labels.

        Returns:
        - stacked_images (numpy array): Images obtained from radar data.
        - labels (numpy array): The combined labels.
        """
        # Concatenate all radar data and labels into single arrays
        stacked_radar_data = np.concatenate(stacked_radar_data, axis=0)  # Shape: (total_time_steps, 1024, 4)
        num_samples, num_features, num_channels = stacked_radar_data.shape

        # Reshape the radar data
        stacked_images = stacked_radar_data.reshape(num_samples, target_shape[0], target_shape[1], num_channels)

        
        return stacked_images, np.concatenate(labels)
