import cv2
import numpy as np
import random

class ImageProcessing:
    def __init__(self, video):
        """
        Class responsible for processing the video frames before passing them to the model.
        :param video: Numpy array with shape (num_frames, height, width, channels)
        """
        self.video = video

    def highlight_players_on_lower_65(self):
        """
        Function highlights the players in a bright, contrasting color (e.g., bright green) in the lower part of the frame.
        """
        processed_video = []

        for frame in self.video:
            height, width, _ = frame.shape

            lower_part = frame[int(height * 0.35):, :]

            ycrcb = cv2.cvtColor(lower_part, cv2.COLOR_BGR2YCrCb)

            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)

            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

            lower_part[skin_mask > 0] = [0, 255, 0]
            lower_part[skin_mask == 0] = [255, 255, 255]

            frame[int(height * 0.35):, :] = lower_part
            processed_video.append(frame)

        self.video = np.array(processed_video)

    def highlight_players_on_upper_35(self):
        """
        Function highlights the players in a bright, contrasting color (e.g., bright green) in the upper part of the frame, 
        modified to avoid detecting yellow colors.
        """
        processed_video = []

        for frame in self.video:
            height, width, _ = frame.shape

            upper_part = frame[:int(height * 0.35), :]

            ycrcb = cv2.cvtColor(upper_part, cv2.COLOR_BGR2YCrCb)

            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)

            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

            yellow_mask = cv2.inRange(ycrcb, np.array([0, 50, 50], dtype=np.uint8), np.array([60, 255, 255], dtype=np.uint8))

            skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(yellow_mask))

            upper_part[skin_mask > 0] = [0, 255, 0]
            upper_part[skin_mask == 0] = [255, 255, 255]

            frame[:int(height * 0.35), :] = upper_part
            processed_video.append(frame)

        self.video = np.array(processed_video)

    def remove_crowd_partial(self):
        """
        Function removes the crowd in the upper 40% of the frame, leaving the lower 60% unchanged.
        """
        processed_video = []

        for frame in self.video:
            height, width, _ = frame.shape
            cutoff = int(height * 0.35)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            upper_mask = np.zeros_like(gray)
            upper_mask[:cutoff, :] = 255
            
            filtered_upper_part = cv2.bitwise_and(edges, upper_mask)
            
            green_mask = np.all(frame == [0, 255, 0], axis=-1)
            
            final_mask = (green_mask & ((filtered_upper_part > 0) | (np.arange(height) >= cutoff)[:, None]))
            
            final_frame = np.full_like(frame, 255)
            final_frame[final_mask] = [0, 255, 0]
            
            processed_video.append(final_frame)
    
        self.video = np.array(processed_video)

    def salt_and_pepper_filter(self, intensity=0.05):
        """
        Function applies a 'salt and pepper' filter to the green [0,255,0] and white [255,255,255] pixels.
        :param intensity: Intensity of the filter, i.e., the percentage of pixels to be changed.
        """
        processed_video = []

        for frame in self.video:
            green_mask = np.all(frame == [0, 255, 0], axis=-1)
            white_mask = np.all(frame == [255, 255, 255], axis=-1)

            total_pixels = frame.shape[0] * frame.shape[1]
            num_changes = int(intensity * total_pixels)

            green_indices = np.where(green_mask)
            white_indices = np.where(white_mask)

            for _ in range(num_changes):
                if green_indices[0].size > 0:
                    idx = random.choice(range(len(green_indices[0])))
                    y, x = green_indices[0][idx], green_indices[1][idx]
                    frame[y, x] = [255, 255, 255]

            for _ in range(num_changes):
                if white_indices[0].size > 0:
                    idx = random.choice(range(len(white_indices[0])))
                    y, x = white_indices[0][idx], white_indices[1][idx]
                    frame[y, x] = [0, 255, 0]

            processed_video.append(frame)

        self.video = np.array(processed_video)

    def sobel_filter(self):
        """
        Function applies the Sobel filter for edge detection in the image.
        """
        processed_video = []

        for frame in self.video:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)

            grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)

            grad = cv2.magnitude(grad_x, grad_y)

            grad = cv2.convertScaleAbs(grad)

            grad_rgb = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

            processed_video.append(grad_rgb)

        self.video = np.array(processed_video)

    def laplacian_filter(self):
        """
        Function applies the Laplacian filter for edge detection in the image.
        """
        processed_video = []

        for frame in self.video:
            laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)
            processed_video.append(laplacian)

        self.video = np.array(processed_video)

    def unsharp_mask(self, alpha=1.5, beta=0.5):
        """
        Function applies the Unsharp Mask filter to sharpen the image.
        :param alpha: Sharpening factor.
        :param beta: Blurring factor.
        """
        processed_video = []

        for frame in self.video:
            blurred = cv2.GaussianBlur(frame, (5, 5), 1.5)
            sharpened = cv2.addWeighted(frame, alpha, blurred, beta, 0)
            processed_video.append(sharpened)

        self.video = np.array(processed_video)

    def canny_filter(self, low_threshold=50, high_threshold=150):
        """
        Function applies the Canny edge detector to the image.
        :param low_threshold: Lower threshold for edge detection.
        :param high_threshold: Upper threshold for edge detection.
        """
        processed_video = []

        for frame in self.video:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(gray_frame, low_threshold, high_threshold)

            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            processed_video.append(edges_rgb)

        self.video = np.array(processed_video)

    def gaussian_blur_upper_30(self, kernel_size=(5, 5)):
        """
        Function applies Gaussian blur to the upper 30% of the image for noise reduction.
        :param kernel_size: Size of the Gaussian kernel, e.g., (5, 5)
        """
        processed_video = []

        for frame in self.video:
            height, width, _ = frame.shape

            upper_part = frame[:int(height * 0.25), :]

            blurred_upper_part = cv2.GaussianBlur(upper_part, kernel_size, 0)

            frame[:int(height * 0.25), :] = blurred_upper_part

            processed_video.append(frame)

        self.video = np.array(processed_video)

    def get_processed_video(self):
        """
        Returns the processed video data.
        """
        return self.video
