# embodied ai - 2024_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15852v7">NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2024-06-30
      | 💬 Accepted by Robotics: Science and Systems (RSS 2024)
    </div>
    <details class="paper-abstract">
      Vision-and-language navigation (VLN) stands as a key research problem of Embodied AI, aiming at enabling agents to navigate in unseen environments following linguistic instructions. In this field, generalization is a long-standing challenge, either to out-of-distribution scenes or from Sim to Real. In this paper, we propose NaVid, a video-based large vision language model (VLM), to mitigate such a generalization gap. NaVid makes the first endeavor to showcase the capability of VLMs to achieve state-of-the-art level navigation performance without any maps, odometers, or depth inputs. Following human instruction, NaVid only requires an on-the-fly video stream from a monocular RGB camera equipped on the robot to output the next-step action. Our formulation mimics how humans navigate and naturally gets rid of the problems introduced by odometer noises, and the Sim2Real gaps from map or depth inputs. Moreover, our video-based approach can effectively encode the historical observations of robots as spatio-temporal contexts for decision making and instruction following. We train NaVid with 510k navigation samples collected from continuous environments, including action-planning and instruction-reasoning samples, along with 763k large-scale web data. Extensive experiments show that NaVid achieves state-of-the-art performance in simulation environments and the real world, demonstrating superior cross-dataset and Sim2Real transfer. We thus believe our proposed VLM approach plans the next step for not only the navigation agents but also this research field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07544v2">Situational Awareness Matters in 3D Vision Language Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-06-26
      | 💬 CVPR 2024. Project Page: https://yunzeman.github.io/situation3d
    </div>
    <details class="paper-abstract">
      Being able to carry out complicated vision language reasoning tasks in 3D space represents a significant milestone in developing household robots and human-centered embodied AI. In this work, we demonstrate that a critical and distinct challenge in 3D vision language reasoning is situational awareness, which incorporates two key components: (1) The autonomous agent grounds its self-location based on a language prompt. (2) The agent answers open-ended questions from the perspective of its calculated position. To address this challenge, we introduce SIG3D, an end-to-end Situation-Grounded model for 3D vision language reasoning. We tokenize the 3D scene into sparse voxel representation and propose a language-grounded situation estimator, followed by a situated question answering module. Experiments on the SQA3D and ScanQA datasets show that SIG3D outperforms state-of-the-art models in situation estimation and question answering by a large margin (e.g., an enhancement of over 30% on situation estimation accuracy). Subsequent analysis corroborates our architectural design choices, explores the distinct functions of visual and textual tokens, and highlights the importance of situational awareness in the domain of 3D question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2406.13807v2">AlanaVLM: A Multimodal Embodied AI Foundation Model for Egocentric Video Understanding</a></div>
    <div class="paper-meta">
      📅 2024-06-24
      | 💬 Code available https://github.com/alanaai/EVUD
    </div>
    <details class="paper-abstract">
      AI personal assistants deployed via robots or wearables require embodied understanding to collaborate with humans effectively. However, current Vision-Language Models (VLMs) primarily focus on third-person view videos, neglecting the richness of egocentric perceptual experience. To address this gap, we propose three key contributions. First, we introduce the Egocentric Video Understanding Dataset (EVUD) for training VLMs on video captioning and question answering tasks specific to egocentric videos. Second, we present AlanaVLM, a 7B parameter VLM trained using parameter-efficient methods on EVUD. Finally, we evaluate AlanaVLM's capabilities on OpenEQA, a challenging benchmark for embodied video question answering. Our model achieves state-of-the-art performance, outperforming open-source models including strong Socratic models using GPT-4 as a planner by 3.6%. Additionally, we outperform Claude 3 and Gemini Pro Vision 1.0 and showcase competitive results compared to Gemini Pro 1.5 and GPT-4V, even surpassing the latter in spatial reasoning. This research paves the way for building efficient VLMs that can be deployed in robots or wearables, leveraging embodied video understanding to collaborate seamlessly with humans in everyday tasks, contributing to the next generation of Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13807v2">AlanaVLM: A Multimodal Embodied AI Foundation Model for Egocentric Video Understanding</a></div>
    <div class="paper-meta">
      📅 2024-06-21
      | 💬 Code available https://github.com/alanaai/EVUD
    </div>
    <details class="paper-abstract">
      AI personal assistants deployed via robots or wearables require embodied understanding to collaborate with humans effectively. However, current Vision-Language Models (VLMs) primarily focus on third-person view videos, neglecting the richness of egocentric perceptual experience. To address this gap, we propose three key contributions. First, we introduce the Egocentric Video Understanding Dataset (EVUD) for training VLMs on video captioning and question answering tasks specific to egocentric videos. Second, we present AlanaVLM, a 7B parameter VLM trained using parameter-efficient methods on EVUD. Finally, we evaluate AlanaVLM's capabilities on OpenEQA, a challenging benchmark for embodied video question answering. Our model achieves state-of-the-art performance, outperforming open-source models including strong Socratic models using GPT-4 as a planner by 3.6%. Additionally, we outperform Claude 3 and Gemini Pro Vision 1.0 and showcase competitive results compared to Gemini Pro 1.5 and GPT-4V, even surpassing the latter in spatial reasoning. This research paves the way for building efficient VLMs that can be deployed in robots or wearables, leveraging embodied video understanding to collaborate seamlessly with humans in everyday tasks, contributing to the next generation of Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05132v2">3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 Project website: https://3d-grand.github.io
    </div>
    <details class="paper-abstract">
      The integration of language and 3D perception is crucial for developing embodied agents and robots that comprehend and interact with the physical world. While large language models (LLMs) have demonstrated impressive language understanding and generation capabilities, their adaptation to 3D environments (3D-LLMs) remains in its early stages. A primary challenge is the absence of large-scale datasets that provide dense grounding between language and 3D scenes. In this paper, we introduce 3D-GRAND, a pioneering large-scale dataset comprising 40,087 household scenes paired with 6.2 million densely-grounded scene-language instructions. Our results show that instruction tuning with 3D-GRAND significantly enhances grounding capabilities and reduces hallucinations in 3D-LLMs. As part of our contributions, we propose a comprehensive benchmark 3D-POPE to systematically evaluate hallucination in 3D-LLMs, enabling fair comparisons among future models. Our experiments highlight a scaling effect between dataset size and 3D-LLM performance, emphasizing the critical role of large-scale 3D-text datasets in advancing embodied AI research. Notably, our results demonstrate early signals for effective sim-to-real transfer, indicating that models trained on large synthetic data can perform well on real-world 3D scans. Through 3D-GRAND and 3D-POPE, we aim to equip the embodied AI community with essential resources and insights, setting the stage for more reliable and better-grounded 3D-LLMs. Project website: https://3d-grand.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18860v2">Empowering Embodied Manipulation: A Bimanual-Mobile Robot Manipulation Dataset for Household Tasks</a></div>
    <div class="paper-meta">
      📅 2024-06-06
    </div>
    <details class="paper-abstract">
      The advancements in embodied AI are increasingly enabling robots to tackle complex real-world tasks, such as household manipulation. However, the deployment of robots in these environments remains constrained by the lack of comprehensive bimanual-mobile robot manipulation data that can be learned. Existing datasets predominantly focus on single-arm manipulation tasks, while the few dual-arm datasets available often lack mobility features, task diversity, comprehensive sensor data, and robust evaluation metrics; they fail to capture the intricate and dynamic nature of household manipulation tasks that bimanual-mobile robots are expected to perform. To overcome these limitations, we propose BRMData, a Bimanual-mobile Robot Manipulation Dataset specifically designed for household applications. BRMData encompasses 10 diverse household tasks, including single-arm and dual-arm tasks, as well as both tabletop and mobile manipulations, utilizing multi-view and depth-sensing data information. Moreover, BRMData features tasks of increasing difficulty, ranging from single-object to multi-object grasping, non-interactive to human-robot interactive scenarios, and rigid-object to flexible-object manipulation, closely simulating real-world household applications. Additionally, we introduce a novel Manipulation Efficiency Score (MES) metric to evaluate both the precision and efficiency of robot manipulation methods in household tasks. We thoroughly evaluate and analyze the performance of advanced robot manipulation learning methods using our BRMData, aiming to drive the development of bimanual-mobile robot manipulation technologies. The dataset is now open-sourced and available at https://embodiedrobot.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2406.00765v1">The Embodied World Model Based on LLM with Visual Information and Prediction-Oriented Prompts</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      In recent years, as machine learning, particularly for vision and language understanding, has been improved, research in embedded AI has also evolved. VOYAGER is a well-known LLM-based embodied AI that enables autonomous exploration in the Minecraft world, but it has issues such as underutilization of visual data and insufficient functionality as a world model. In this research, the possibility of utilizing visual data and the function of LLM as a world model were investigated with the aim of improving the performance of embodied AI. The experimental results revealed that LLM can extract necessary information from visual data, and the utilization of the information improves its performance as a world model. It was also suggested that devised prompts could bring out the LLM's function as a world model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2406.00375v1">Teledrive: An Embodied AI based Telepresence System</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 Accepted in Journal of Intelligent Robotic System
    </div>
    <details class="paper-abstract">
      This article presents Teledrive, a telepresence robotic system with embodied AI features that empowers an operator to navigate the telerobot in any unknown remote place with minimal human intervention. We conceive Teledrive in the context of democratizing remote care-giving for elderly citizens as well as for isolated patients, affected by contagious diseases. In particular, this paper focuses on the problem of navigating to a rough target area (like bedroom or kitchen) rather than pre-specified point destinations. This ushers in a unique AreaGoal based navigation feature, which has not been explored in depth in the contemporary solutions. Further, we describe an edge computing-based software system built on a WebRTC-based communication framework to realize the aforementioned scheme through an easy-to-use speech-based human-robot interaction. Moreover, to enhance the ease of operation for the remote caregiver, we incorporate a person following feature, whereby a robot follows a person on the move in its premises as directed by the operator. Moreover, the system presented is loosely coupled with specific robot hardware, unlike the existing solutions. We have evaluated the efficacy of the proposed system through baseline experiments, user study, and real-life deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00765v1">The Embodied World Model Based on LLM with Visual Information and Prediction-Oriented Prompts</a></div>
    <div class="paper-meta">
      📅 2024-06-02
    </div>
    <details class="paper-abstract">
      In recent years, as machine learning, particularly for vision and language understanding, has been improved, research in embedded AI has also evolved. VOYAGER is a well-known LLM-based embodied AI that enables autonomous exploration in the Minecraft world, but it has issues such as underutilization of visual data and insufficient functionality as a world model. In this research, the possibility of utilizing visual data and the function of LLM as a world model were investigated with the aim of improving the performance of embodied AI. The experimental results revealed that LLM can extract necessary information from visual data, and the utilization of the information improves its performance as a world model. It was also suggested that devised prompts could bring out the LLM's function as a world model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00375v1">Teledrive: An Embodied AI based Telepresence System</a></div>
    <div class="paper-meta">
      📅 2024-06-01
      | 💬 Accepted in Journal of Intelligent Robotic System
    </div>
    <details class="paper-abstract">
      This article presents Teledrive, a telepresence robotic system with embodied AI features that empowers an operator to navigate the telerobot in any unknown remote place with minimal human intervention. We conceive Teledrive in the context of democratizing remote care-giving for elderly citizens as well as for isolated patients, affected by contagious diseases. In particular, this paper focuses on the problem of navigating to a rough target area (like bedroom or kitchen) rather than pre-specified point destinations. This ushers in a unique AreaGoal based navigation feature, which has not been explored in depth in the contemporary solutions. Further, we describe an edge computing-based software system built on a WebRTC-based communication framework to realize the aforementioned scheme through an easy-to-use speech-based human-robot interaction. Moreover, to enhance the ease of operation for the remote caregiver, we incorporate a person following feature, whereby a robot follows a person on the move in its premises as directed by the operator. Moreover, the system presented is loosely coupled with specific robot hardware, unlike the existing solutions. We have evaluated the efficacy of the proposed system through baseline experiments, user study, and real-life deployment.
    </details>
</div>
