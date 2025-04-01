import cv2
import h5py
import os
import numpy as np

class DataPreparation:
    def __init__(self, video_folder_left, video_folder_right, image_size=(224, 224), max_frames=180):
        """
        Initializes the DataPreparation class for preparing video data for 3D CNN.
        
        :param video_folder_left: Folder containing videos where Left team scored.
        :param video_folder_right: Folder containing videos where Right team scored.
        :param image_size: The size to which each frame will be resized (default is 224x224).
        :param max_frames: Maximum number of frames to extract from each video (default is 300).
        """
        self.video_folder_left = video_folder_left
        self.video_folder_right = video_folder_right
        self.image_size = image_size
        self.max_frames = max_frames
        self.data = np.array([])
        self.labels = np.array([])

    def extract_all_videos(self):
        """
        Extracts all frames from videos in both Left and Right folders and prepares data for classification.
        The method returns two arrays: one for video data and one for labels.
        """
        video_data = []
        labels = []

        # Process videos from the Left folder
        for video_file in os.listdir(self.video_folder_left):
            if video_file.endswith(".mp4"):  # You can add other formats if needed
                video_path = os.path.join(self.video_folder_left, video_file)
                print(f"Processing video: {video_path}")

                # Extract frames for this video
                video_frames = self._extract_frames_from_video(video_path)
                video_data.append(video_frames)
                labels.append(0)  # Label 0 for Left team scoring

        # Process videos from the Right folder
        for video_file in os.listdir(self.video_folder_right):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(self.video_folder_right, video_file)
                print(f"Processing video: {video_path}")

                # Extract frames for this video
                video_frames = self._extract_frames_from_video(video_path)
                video_data.append(video_frames)
                labels.append(1)  # Label 1 for Right team scoring

        self.data = np.array(video_data)
        self.labels = np.array(labels)

    def _extract_frames_from_video(self, video_path):
        """
        Extracts frames from a single video and resizes them.
        The video frames are collected into a numpy array.

        :param video_path: Path to the video file.
        :return: A numpy array with shape (num_frames, height, width, channels).
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, self.image_size)
            frames.append(frame_resized)

        cap.release()

        if len(frames) < self.max_frames:
            print(len(frames))
            last_frame = frames[-1]
            while len(frames) < self.max_frames:
                frames.append(last_frame)

        elif len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = [frames[i] for i in indices] 

        return np.array(frames)


    def save_video(self, frames, output_path, fps=30):
        """
        Creates .mp4 file based on frames.
        
        :param frames: List of frames.
        :param output_path: path to output file.
        :param fps: frames per second.
        """
        if len(frames) == 0:
            print("No frames")
            return
        
        height, width, _ = frames[0].shape 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Zapisano wideo: {output_path}")

    def save_data_hdf5(self, file_path):
        """
        save data to .h5.
        
        :param video_data: transformed video file.
        :param labels: labels.
        :param file_path: file to patj .h5.
        """
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('videos', data = self.data)
            f.create_dataset('labels', data = self.labels)
        print(f"Saved data to {file_path}")

    def load_data_hdf5(self, file_path):
        """
        loda data from file .h5.
        
        :param file_path: path to file .h5.
        :return: video_data, labels
        """
        with h5py.File(file_path, 'r') as f:
            video_data = np.array(f['videos'])
            labels = np.array(f['labels'])
        self.data = video_data
        self.labels = labels


# Example usage:
if __name__ == "__main__":
    video_folder_left = "C:\Studia\Magisterka\Projekt\Left_test"
    video_folder_right = "C:\Studia\Magisterka\Projekt\Right_test"

    data_preparation = DataPreparation(video_folder_left, video_folder_right)

    #data_preparation.extract_all_videos()
    #data_preparation.load_data_hdf5("prepared_data.h5")
    
    print(f"Shape of video data: {data_preparation.data.shape}")
    print(f"Shape of labels: {data_preparation.labels.shape}")
    #data_preparation.save_data_hdf5("prepared_data.h5")

   