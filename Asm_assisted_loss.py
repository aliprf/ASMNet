class ASMLoss:
    def __init__(self, dataset_name, accuracy):
        self.dataset_name = dataset_name
        self.accuracy = accuracy

    def calculate_pose_loss(self, x_pr, x_gt):
        pass

    def calculate_landmark_ASM_assisted_loss(self, landmark_pr, landmark_gt):
        pass


