from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video('input_videos/test(33).mp4')

    #initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_subs.pkl')

    save_video(video_frames, 'output_videos/output_video.avi')



if __name__ == '__main__':
    main()