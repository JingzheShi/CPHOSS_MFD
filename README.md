# CPHOSS_MFD
A repo for math formula detection used in the AI Grading project called Centralized Physics Olympiad Scoring System.


# Summary
This repository uses the same model and based on https://github.com/Yuxiang1995/ICDAR2021_MFD. The contribution of this repository can be concluded as following:

Previous MFD (Math Formula Detection) tasks mainly focus on detecting math formulas in essays. In these essays the formulas may be embedded in paragraphs of texts. However in our case of AI Grading for Physics Olympiad, a common case is that there are many formulas lying on the answer sheet with no or very few texts. In other word this is not similar to the training set provided by the essay dataset. Thus models trained on the essay dataset perform poorly in our test case.

The repo's contribution is that it **provides a way to generate a new dataset suitable for MFD used in grading answer sheets based on the essay dataset**. The main idea is like **CAP** (Cut and Paste), an Augmentation method usually used in 3D-object detection. That is, we cut the formulas in the essay dataset, then paste them at random onto a white paper with scaling or rotation as an augmentation method. Models trained on this generated new dataset can perform very well in our test case.

#Examples
Here we provide some examples, showing the effectness of our generated dataset and the adaptation ability of models trained on this dataset to different conditions.

# Citations
```shell
@article{zhong20211st,
  title={1st Place Solution for ICDAR 2021 Competition on Mathematical Formula Detection},
  author={Zhong, Yuxiang and Qi, Xianbiao and Li, Shanjun and Gu, Dengyi and Chen, Yihao and Ning, Peiyang and Xiao, Rong},
  journal={arXiv preprint arXiv:2107.05534},
  year={2021}
}
