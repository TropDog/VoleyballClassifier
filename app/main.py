from DataPreparation import DataPreparation
from ImageProcessing import ImageProcessing

video_folder_left = "C:\\Studia\\Magisterka\\Projekt\\Left_test"
video_folder_right = "C:\\Studia\\Magisterka\\Projekt\\Right_test"
data_preparation = DataPreparation(video_folder_left, video_folder_right)
data_preparation.extract_all_videos()


for i in range(len(data_preparation.data)):
    img_proc = ImageProcessing(data_preparation.data[i])
    img_proc.highlight_players_on_lower_65()
    img_proc.highlight_players_on_upper_35()
    img_proc.remove_crowd_partial()
    img_proc.gaussian_blur_upper_30((11,11))
    img_proc.unsharp_mask()
    img_proc.sobel_filter()


    video = img_proc.get_processed_video()  
    data_preparation.data[i] = video

for i in range(len(data_preparation.data)):
    data_preparation.save_video(data_preparation.data[i], f"final_{i}.mp4")