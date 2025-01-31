# llm - 2024_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11453v3">MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Scientific data visualization plays a crucial role in research by enabling the direct display of complex information and assisting researchers in identifying implicit patterns. Despite its importance, the use of Large Language Models (LLMs) for scientific data visualization remains rather unexplored. In this study, we introduce MatPlotAgent, an efficient model-agnostic LLM agent framework designed to automate scientific data visualization tasks. Leveraging the capabilities of both code LLMs and multi-modal LLMs, MatPlotAgent consists of three core modules: query understanding, code generation with iterative debugging, and a visual feedback mechanism for error correction. To address the lack of benchmarks in this field, we present MatPlotBench, a high-quality benchmark consisting of 100 human-verified test cases. Additionally, we introduce a scoring approach that utilizes GPT-4V for automatic evaluation. Experimental results demonstrate that MatPlotAgent can improve the performance of various LLMs, including both commercial and open-source models. Furthermore, the proposed evaluation method shows a strong correlation with human-annotated scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.13812v2">Dysen-VDM: Empowering Dynamics-aware Text-to-Video Diffusion with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 CVPR 2024
    </div>
    <details class="paper-abstract">
      Text-to-video (T2V) synthesis has gained increasing attention in the community, in which the recently emerged diffusion models (DMs) have promisingly shown stronger performance than the past approaches. While existing state-of-the-art DMs are competent to achieve high-resolution video generation, they may largely suffer from key limitations (e.g., action occurrence disorders, crude video motions) with respect to the intricate temporal dynamics modeling, one of the crux of video synthesis. In this work, we investigate strengthening the awareness of video dynamics for DMs, for high-quality T2V generation. Inspired by human intuition, we design an innovative dynamic scene manager (dubbed as Dysen) module, which includes (step-1) extracting from input text the key actions with proper time-order arrangement, (step-2) transforming the action schedules into the dynamic scene graph (DSG) representations, and (step-3) enriching the scenes in the DSG with sufficient and reasonable details. Taking advantage of the existing powerful LLMs (e.g., ChatGPT) via in-context learning, Dysen realizes (nearly) human-level temporal dynamics understanding. Finally, the resulting video DSG with rich action scene details is encoded as fine-grained spatio-temporal features, integrated into the backbone T2V DM for video generating. Experiments on popular T2V datasets suggest that our Dysen-VDM consistently outperforms prior arts with significant margins, especially in scenarios with complex actions. Codes at https://haofei.vip/Dysen-VDM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.05903v2">Establishing Performance Baselines in Fine-Tuning, Retrieval-Augmented Generation and Soft-Prompting for Non-Specialist LLM Users</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 10 pages, LaTeX; typos corrected, using the correct term 'system prompting' instead of 'soft prompting'
    </div>
    <details class="paper-abstract">
      Research into methods for improving the performance of large language models (LLMs) through fine-tuning, retrieval-augmented generation (RAG) and soft-prompting has tended to focus on the use of highly technical or high-cost techniques, making many of the newly discovered approaches comparatively inaccessible to non-technical users. In this paper we tested an unmodified version of GPT 3.5, a fine-tuned version, and the same unmodified model when given access to a vectorised RAG database, both in isolation and in combination with a basic, non-algorithmic soft prompt. In each case we tested the model's ability to answer a set of 100 questions relating primarily to events that occurred after September 2021 (the point at which GPT 3.5's training data set ends). We found that if commercial platforms are used and default settings are applied with no iteration in order to establish a baseline set of outputs, a fine-tuned model outperforms GPT 3.5 Turbo, while the RAG approach out-performed both. The application of a soft prompt significantly improved the performance of each approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12596v1">Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 Findings of NAACL 2024
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) are achieving increasingly strong performance on multimodal tasks. However, reasoning capabilities remain limited particularly for smaller VLMs, while those of large-language models (LLMs) have seen numerous improvements. We propose a technique to transfer capabilities from LLMs to VLMs. On the recently introduced ChartQA, our method obtains state-of-the-art performance when applied on the PaLI3-5B VLM by \citet{chen2023pali3}, while also enabling much better performance on PlotQA and FigureQA. We first improve the chart representation by continuing the pre-training stage using an improved version of the chart-to-table translation task by \citet{liu2023deplot}. We then propose constructing a 20x larger dataset than the original training set. To improve general reasoning capabilities and improve numerical operations, we synthesize reasoning traces using the table representation of charts. Lastly, our model is fine-tuned using the multitask loss introduced by \citet{hsieh2023distilling}. Our variant ChartPaLI-5B outperforms even 10x larger models such as PaLIX-55B without using an upstream OCR system, while keeping inference time constant compared to the PaLI3-5B baseline. When rationales are further refined with a simple program-of-thought prompt \cite{chen2023program}, our model outperforms the recently introduced Gemini Ultra and GPT-4V.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12532v1">UniBind: LLM-Augmented Unified and Balanced Representation Space to Bind Them All</a></div>
    <div class="paper-meta">
      📅 2024-03-19
      | 💬 Accepted to CVPR2024
    </div>
    <details class="paper-abstract">
      We present UniBind, a flexible and efficient approach that learns a unified representation space for seven diverse modalities -- images, text, audio, point cloud, thermal, video, and event data. Existing works, eg., ImageBind, treat the image as the central modality and build an image-centered representation space; however, the space may be sub-optimal as it leads to an unbalanced representation space among all modalities. Moreover, the category names are directly used to extract text embeddings for the downstream tasks, making it hardly possible to represent the semantics of multi-modal data. The 'out-of-the-box' insight of our UniBind is to make the alignment center modality-agnostic and further learn a unified and balanced representation space, empowered by the large language models (LLMs). UniBind is superior in its flexible application to all CLIP-style models and delivers remarkable performance boosts. To make this possible, we 1) construct a knowledge base of text embeddings with the help of LLMs and multi-modal LLMs; 2) adaptively build LLM-augmented class-wise embedding center on top of the knowledge base and encoded visual embeddings; 3) align all the embeddings to the LLM-augmented embedding center via contrastive learning to achieve a unified and balanced representation space. UniBind shows strong zero-shot recognition performance gains over prior arts by an average of 6.36%. Finally, we achieve new state-of-the-art performance, eg., a 6.75% gain on ImageNet, on the multi-modal fine-tuning setting while reducing 90% of the learnable parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12415v1">VisionGPT: LLM-Assisted Real-Time Anomaly Detection for Safe Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2024-03-19
    </div>
    <details class="paper-abstract">
      This paper explores the potential of Large Language Models(LLMs) in zero-shot anomaly detection for safe visual navigation. With the assistance of the state-of-the-art real-time open-world object detection model Yolo-World and specialized prompts, the proposed framework can identify anomalies within camera-captured frames that include any possible obstacles, then generate concise, audio-delivered descriptions emphasizing abnormalities, assist in safe visual navigation in complex circumstances. Moreover, our proposed framework leverages the advantages of LLMs and the open-vocabulary object detection model to achieve the dynamic scenario switch, which allows users to transition smoothly from scene to scene, which addresses the limitation of traditional visual navigation. Furthermore, this paper explored the performance contribution of different prompt components, provided the vision for future improvement in visual accessibility, and paved the way for LLMs in video anomaly detection and vision-language understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.13027v1">Towards Better Statistical Understanding of Watermarking LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-19
    </div>
    <details class="paper-abstract">
      In this paper, we study the problem of watermarking large language models (LLMs). We consider the trade-off between model distortion and detection ability and formulate it as a constrained optimization problem based on the green-red algorithm of Kirchenbauer et al. (2023a). We show that the optimal solution to the optimization problem enjoys a nice analytical property which provides a better understanding and inspires the algorithm design for the watermarking process. We develop an online dual gradient ascent watermarking algorithm in light of this optimization formulation and prove its asymptotic Pareto optimality between model distortion and detection ability. Such a result guarantees an averaged increased green list probability and henceforth detection ability explicitly (in contrast to previous results). Moreover, we provide a systematic discussion on the choice of the model distortion metrics for the watermarking problem. We justify our choice of KL divergence and present issues with the existing criteria of ``distortion-free'' and perplexity. Finally, we empirically evaluate our algorithms on extensive datasets against benchmark algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12316v1">OpenEval: Benchmarking Chinese LLMs across Capability, Alignment and Safety</a></div>
    <div class="paper-meta">
      📅 2024-03-18
    </div>
    <details class="paper-abstract">
      The rapid development of Chinese large language models (LLMs) poses big challenges for efficient LLM evaluation. While current initiatives have introduced new benchmarks or evaluation platforms for assessing Chinese LLMs, many of these focus primarily on capabilities, usually overlooking potential alignment and safety issues. To address this gap, we introduce OpenEval, an evaluation testbed that benchmarks Chinese LLMs across capability, alignment and safety. For capability assessment, we include 12 benchmark datasets to evaluate Chinese LLMs from 4 sub-dimensions: NLP tasks, disciplinary knowledge, commonsense reasoning and mathematical reasoning. For alignment assessment, OpenEval contains 7 datasets that examines the bias, offensiveness and illegalness in the outputs yielded by Chinese LLMs. To evaluate safety, especially anticipated risks (e.g., power-seeking, self-awareness) of advanced LLMs, we include 6 datasets. In addition to these benchmarks, we have implemented a phased public evaluation and benchmark update strategy to ensure that OpenEval is in line with the development of Chinese LLMs or even able to provide cutting-edge benchmark datasets to guide the development of Chinese LLMs. In our first public evaluation, we have tested a range of Chinese LLMs, spanning from 7B to 72B parameters, including both open-source and proprietary models. Evaluation results indicate that while Chinese LLMs have shown impressive performance in certain tasks, more attention should be directed towards broader aspects such as commonsense reasoning, alignment, and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.14459v2">Bridging the Gulf of Envisioning: Cognitive Design Challenges in LLM Interfaces</a></div>
    <div class="paper-meta">
      📅 2024-03-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit dynamic capabilities and appear to comprehend complex and ambiguous natural language prompts. However, calibrating LLM interactions is challenging for interface designers and end-users alike. A central issue is our limited grasp of how human cognitive processes begin with a goal and form intentions for executing actions, a blindspot even in established interaction models such as Norman's gulfs of execution and evaluation. To address this gap, we theorize how end-users 'envision' translating their goals into clear intentions and craft prompts to obtain the desired LLM response. We define a process of Envisioning by highlighting three misalignments: (1) knowing whether LLMs can accomplish the task, (2) how to instruct the LLM to do the task, and (3) how to evaluate the success of the LLM's output in meeting the goal. Finally, we make recommendations to narrow the envisioning gulf in human-LLM interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11863v1">Context-aware LLM-based Safe Control Against Latent Risks</a></div>
    <div class="paper-meta">
      📅 2024-03-18
    </div>
    <details class="paper-abstract">
      It is challenging for autonomous control systems to perform complex tasks in the presence of latent risks. Motivated by this challenge, this paper proposes an integrated framework that involves Large Language Models (LLMs), stochastic gradient descent (SGD), and optimization-based control. In the first phrase, the proposed framework breaks down complex tasks into a sequence of smaller subtasks, whose specifications account for contextual information and latent risks. In the second phase, these subtasks and their parameters are refined through a dual process involving LLMs and SGD. LLMs are used to generate rough guesses and failure explanations, and SGD is used to fine-tune parameters. The proposed framework is tested using simulated case studies of robots and vehicles. The experiments demonstrate that the proposed framework can mediate actions based on the context and latent risks and learn complex behaviors efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11810v1">Metaphor Understanding Challenge Dataset for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-18
    </div>
    <details class="paper-abstract">
      Metaphors in natural language are a reflection of fundamental cognitive processes such as analogical reasoning and categorisation, and are deeply rooted in everyday communication. Metaphor understanding is therefore an essential task for large language models (LLMs). We release the Metaphor Understanding Challenge Dataset (MUNCH), designed to evaluate the metaphor understanding capabilities of LLMs. The dataset provides over 10k paraphrases for sentences containing metaphor use, as well as 1.5k instances containing inapt paraphrases. The inapt paraphrases were carefully selected to serve as control to determine whether the model indeed performs full metaphor interpretation or rather resorts to lexical similarity. All apt and inapt paraphrases were manually annotated. The metaphorical sentences cover natural metaphor uses across 4 genres (academic, news, fiction, and conversation), and they exhibit different levels of novelty. Experiments with LLaMA and GPT-3.5 demonstrate that MUNCH presents a challenging task for LLMs. The dataset is freely accessible at https://github.com/xiaoyuisrain/metaphor-understanding-challenge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11805v1">LLM as a System Service on Mobile Devices</a></div>
    <div class="paper-meta">
      📅 2024-03-18
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Being more powerful and intrusive into user-device interactions, LLMs are eager for on-device execution to better preserve user privacy. In this work, we propose a new paradigm of mobile AI: LLM as a system service on mobile devices (LLMaaS). Unlike traditional DNNs that execute in a stateless manner, such a system service is stateful: LLMs execution often needs to maintain persistent states (mainly KV cache) across multiple invocations. To minimize the LLM context switching overhead under tight device memory budget, this work presents LLMS, which decouples the memory management of app and LLM contexts with a key idea of fine-grained, chunk-wise, globally-optimized KV cache compression and swapping. By fully leveraging KV cache's unique characteristics, it proposes three novel techniques: (1) Tolerance-Aware Compression: it compresses chunks based on their measured accuracy tolerance to compression. (2) IO-Recompute Pipelined Loading: it introduces recompute to swapping-in for acceleration. (3) Chunk Lifecycle Management: it optimizes the memory activities of chunks with an ahead-of-time swapping-out and an LCTRU (Least Compression-Tolerable and Recently-Used) queue based eviction. In evaluations conducted on well-established traces and various edge devices, \sys reduces context switching latency by up to 2 orders of magnitude when compared to competitive baseline solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06838v2">ACFIX: Guiding LLMs with Mined Common RBAC Practices for Context-Aware Repair of Access Control Vulnerabilities in Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2024-03-18
      | 💬 This is a technical report from Nanyang Technological University
    </div>
    <details class="paper-abstract">
      Smart contracts are susceptible to various security issues, among which access control (AC) vulnerabilities are particularly critical. While existing research has proposed multiple detection tools, the automatic and appropriate repair of AC vulnerabilities in smart contracts remains a challenge. Unlike commonly supported vulnerability types by existing repair tools, such as reentrancy, which are usually fixed by template-based approaches, the main obstacle of AC lies in identifying the appropriate roles or permissions amid a long list of non-AC-related source code to generate proper patch code, a task that demands human-level intelligence. Leveraging recent advancements in large language models (LLMs), we employ the state-of-the-art GPT-4 model and enhance it with a novel approach called ACFIX. The key insight is that we can mine common AC practices for major categories of code functionality and use them to guide LLMs in fixing code with similar functionality. To this end, ACFIX involves both offline and online phases. First, during the offline phase, ACFIX mines a taxonomy of common Role-based Access Control (RBAC) practices from 344,251 on-chain contracts, categorizing 49 role-permission pairs from the top 1,000 pairs mined. Second, during the online phase, ACFIX tracks AC-related elements across the contract and uses this context information along with a Chain-of-Thought pipeline to guide LLMs in identifying the most appropriate role-permission pair for the subject contract and subsequently generating a suitable patch. This patch will then undergo a validity and effectiveness check. To evaluate ACFIX, we built the first benchmark dataset of 118 real-world AC vulnerabilities, and our evaluation revealed that ACFIX successfully repaired 94.92% of them. This represents a significant improvement compared to the baseline GPT-4, which achieved only 52.54%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.10900v2">MotionGPT: Finetuned LLMs Are General-Purpose Motion Generators</a></div>
    <div class="paper-meta">
      📅 2024-03-18
      | 💬 18 pages, 8 figures, accepted by AAAI 2024
    </div>
    <details class="paper-abstract">
      Generating realistic human motion from given action descriptions has experienced significant advancements because of the emerging requirement of digital humans. While recent works have achieved impressive results in generating motion directly from textual action descriptions, they often support only a single modality of the control signal, which limits their application in the real digital human industry. This paper presents a Motion General-Purpose generaTor (MotionGPT) that can use multimodal control signals, e.g., text and single-frame poses, for generating consecutive human motions by treating multimodal signals as special input tokens in large language models (LLMs). Specifically, we first quantize multimodal control signals into discrete codes and then formulate them in a unified prompt instruction to ask the LLMs to generate the motion answer. Our MotionGPT demonstrates a unified human motion generation model with multimodal control signals by tuning a mere 0.4% of LLM parameters. To the best of our knowledge, MotionGPT is the first method to generate human motion by multimodal control signals, which we hope can shed light on this new direction. Visit our webpage at https://qiqiapink.github.io/MotionGPT/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11446v1">LLM Guided Evolution -- The Automation of Models Advancing Models</a></div>
    <div class="paper-meta">
      📅 2024-03-18
    </div>
    <details class="paper-abstract">
      In the realm of machine learning, traditional model development and automated approaches like AutoML typically rely on layers of abstraction, such as tree-based or Cartesian genetic programming. Our study introduces "Guided Evolution" (GE), a novel framework that diverges from these methods by utilizing Large Language Models (LLMs) to directly modify code. GE leverages LLMs for a more intelligent, supervised evolutionary process, guiding mutations and crossovers. Our unique "Evolution of Thought" (EoT) technique further enhances GE by enabling LLMs to reflect on and learn from the outcomes of previous mutations. This results in a self-sustaining feedback loop that augments decision-making in model evolution. GE maintains genetic diversity, crucial for evolutionary algorithms, by leveraging LLMs' capability to generate diverse responses from expertly crafted prompts and modulate model temperature. This not only accelerates the evolution process but also injects expert like creativity and insight into the process. Our application of GE in evolving the ExquisiteNetV2 model demonstrates its efficacy: the LLM-driven GE autonomously produced variants with improved accuracy, increasing from 92.52% to 93.34%, without compromising model compactness. This underscores the potential of LLMs to accelerate the traditional model design pipeline, enabling models to autonomously evolve and enhance their own designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11490v5">LLM-CXR: Instruction-Finetuned LLM for CXR Image Understanding and Generation</a></div>
    <div class="paper-meta">
      📅 2024-03-18
      | 💬 21 pages, 8 figures; ICLR 2024 (poster)
    </div>
    <details class="paper-abstract">
      Following the impressive development of LLMs, vision-language alignment in LLMs is actively being researched to enable multimodal reasoning and visual IO. This direction of research is particularly relevant to medical imaging because medical image analysis and generation consist of reasoning based on a combination of visual features and prior knowledge. Many recent works have focused on training adapter networks that serve as an information bridge between image processing networks and LLMs; but presumably, in order to achieve maximum reasoning potential of LLMs on visual information as well, visual and language features should be allowed to interact more freely. This is especially important in the medical domain because understanding and generating medical images such as chest X-rays (CXR) require not only accurate visual and language-based reasoning but also a more intimate mapping between the two modalities. Thus, taking inspiration from previous work on the transformer and VQ-GAN combination for bidirectional image and text generation, we build upon this approach and develop a method for instruction-tuning an LLM pre-trained only on text to gain vision-language capabilities for medical images. Specifically, we leverage a pretrained LLM's existing question-answering and instruction-following abilities to teach it to understand visual inputs by instructing it to answer questions about image inputs and, symmetrically, output both text and image responses appropriate to a given query by tuning the LLM with diverse tasks that encompass image-based text-generation and text-based image-generation. We show that our model, LLM-CXR, trained in this approach shows better image-text alignment in both CXR understanding and generation tasks while being smaller in size compared to previously developed models that perform a narrower range of tasks. The code is at https://github.com/hyn2028/llm-cxr.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11439v1">StyleChat: Learning Recitation-Augmented Memory in LLMs for Stylized Dialogue Generation</a></div>
    <div class="paper-meta">
      📅 2024-03-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate superior performance in generative scenarios and have attracted widespread attention. Among them, stylized dialogue generation is essential in the context of LLMs for building intelligent and engaging dialogue agent. However the ability of LLMs is data-driven and limited by data bias, leading to poor performance on specific tasks. In particular, stylized dialogue generation suffers from a severe lack of supervised data. Furthermore, although many prompt-based methods have been proposed to accomplish specific tasks, their performance in complex real-world scenarios involving a wide variety of dialog styles further enhancement. In this work, we first introduce a stylized dialogue dataset StyleEval with 38 styles by leveraging the generative power of LLMs comprehensively, which has been carefully constructed with rigorous human-led quality control. Based on this, we propose the stylized dialogue framework StyleChat via recitation-augmented memory strategy and multi-task style learning strategy to promote generalization ability. To evaluate the effectiveness of our approach, we created a test benchmark that included both a generation task and a choice task to comprehensively evaluate trained models and assess whether styles and preferences are remembered and understood. Experimental results show that our proposed framework StyleChat outperforms all the baselines and helps to break the style boundary of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11421v1">FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines</a></div>
    <div class="paper-meta">
      📅 2024-03-18
      | 💬 15 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Cost of serving large language models (LLM) is high, but the expensive and scarce GPUs are poorly efficient when generating tokens sequentially, unless the batch of sequences is enlarged. However, the batch size is limited by some constantly reused intermediate results, namely KV-Cache. They occupy too much memory to fit more sequences into a GPU simultaneously. While they could be offloaded to host memory, the CPU-GPU bandwidth is an inevitable bottleneck. We find a way to decompose the transformer models into two parts of different characteristics, one of which includes the memory-bound KV-Cache accessing. Our key insight is that the aggregated memory capacity, bandwidth, and computing power of CPUs across multiple nodes is an efficient option to process this part. Performance improvement comes from reduced data transmission overhead and boosted GPU throughput to process the other model part. Moreover, we address efficiency challenges brought by heterogeneity at both temporal and inter-device scopes using scheduling and performance modeling techniques. Evaluation results show that our system achieves 1.88x - 5.04x the throughput of vLLM when serving modern LLMs with the same GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01557v5">SmartPlay: A Benchmark for LLMs as Intelligent Agents</a></div>
    <div class="paper-meta">
      📅 2024-03-17
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have demonstrated great potential toward intelligent agents and next-gen automation, but there currently lacks a systematic benchmark for evaluating LLMs' abilities as agents. We introduce SmartPlay: both a challenging benchmark and a methodology for evaluating LLMs as agents. SmartPlay consists of 6 different games, including Rock-Paper-Scissors, Tower of Hanoi, Minecraft. Each game features a unique setting, providing up to 20 evaluation settings and infinite environment variations. Each game in SmartPlay uniquely challenges a subset of 9 important capabilities of an intelligent LLM agent, including reasoning with object dependencies, planning ahead, spatial reasoning, learning from history, and understanding randomness. The distinction between the set of capabilities each game test allows us to analyze each capability separately. SmartPlay serves not only as a rigorous testing ground for evaluating the overall performance of LLM agents but also as a road-map for identifying gaps in current methodologies. We release our benchmark at github.com/Microsoft/SmartPlay
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11368v1">Driving Style Alignment for LLM-powered Driver Agent</a></div>
    <div class="paper-meta">
      📅 2024-03-17
    </div>
    <details class="paper-abstract">
      Recently, LLM-powered driver agents have demonstrated considerable potential in the field of autonomous driving, showcasing human-like reasoning and decision-making abilities.However, current research on aligning driver agent behaviors with human driving styles remains limited, partly due to the scarcity of high-quality natural language data from human driving behaviors.To address this research gap, we propose a multi-alignment framework designed to align driver agents with human driving styles through demonstrations and feedback. Notably, we construct a natural language dataset of human driver behaviors through naturalistic driving experiments and post-driving interviews, offering high-quality human demonstrations for LLM alignment. The framework's effectiveness is validated through simulation experiments in the CARLA urban traffic simulator and further corroborated by human evaluations. Our research offers valuable insights into designing driving agents with diverse driving styles.The implementation of the framework and details of the dataset can be found at the link.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11317v1">Few-Shot VQA with Frozen LLMs: A Tale of Two Approaches</a></div>
    <div class="paper-meta">
      📅 2024-03-17
    </div>
    <details class="paper-abstract">
      Two approaches have emerged to input images into large language models (LLMs). The first is to caption images into natural language. The second is to map image feature embeddings into the domain of the LLM and pass the mapped embeddings directly to the LLM. The majority of recent few-shot multimodal work reports performance using architectures that employ variations of one of these two approaches. But they overlook an important comparison between them. We design a controlled and focused experiment to compare these two approaches to few-shot visual question answering (VQA) with LLMs. Our findings indicate that for Flan-T5 XL, a 3B parameter LLM, connecting visual embeddings directly to the LLM embedding space does not guarantee improved performance over using image captions. In the zero-shot regime, we find using textual image captions is better. In the few-shot regimes, how the in-context examples are selected determines which is better.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08754v4">Tokenizer Choice For LLM Training: Negligible or Crucial?</a></div>
    <div class="paper-meta">
      📅 2024-03-17
    </div>
    <details class="paper-abstract">
      The recent success of Large Language Models (LLMs) has been predominantly driven by curating the training dataset composition, scaling of model architectures and dataset sizes and advancements in pretraining objectives, leaving tokenizer influence as a blind spot. Shedding light on this underexplored area, we conduct a comprehensive study on the influence of tokenizer choice on LLM downstream performance by training 24 mono- and multilingual LLMs at a 2.6B parameter scale, ablating different tokenizer algorithms and parameterizations. Our studies highlight that the tokenizer choice can significantly impact the model's downstream performance and training costs. In particular, we find that the common tokenizer evaluation metrics fertility and parity are not always predictive of model downstream performance, rendering these metrics a questionable proxy for the model's downstream performance. Furthermore, we show that multilingual tokenizers trained on the five most frequent European languages require vocabulary size increases of factor three in comparison to English. While English-centric tokenizers have been applied to the training of multi-lingual LLMs in the past, we find that this approach results in a severe downstream performance degradation and additional training costs of up to 68%, due to an inefficient tokenization vocabulary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.10855v3">LatEval: An Interactive LLMs Evaluation Benchmark with Incomplete Information from Lateral Thinking Puzzles</a></div>
    <div class="paper-meta">
      📅 2024-03-17
      | 💬 Accepted by LREC-COLING 2024
    </div>
    <details class="paper-abstract">
      With the continuous evolution and refinement of LLMs, they are endowed with impressive logical reasoning or vertical thinking capabilities. But can they think out of the box? Do they possess proficient lateral thinking abilities? Following the setup of Lateral Thinking Puzzles, we propose a novel evaluation benchmark, LatEval, which assesses the model's lateral thinking within an interactive framework. In our benchmark, we challenge LLMs with 2 aspects: the quality of questions posed by the model and the model's capability to integrate information for problem-solving. We find that nearly all LLMs struggle with employing lateral thinking during interactions. For example, even the most advanced model, GPT-4, exhibits the advantage to some extent, yet still maintain a noticeable gap when compared to human. This evaluation benchmark provides LLMs with a highly challenging and distinctive task that is crucial to an effective AI assistant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11152v1">Evaluation Ethics of LLMs in Legal Domain</a></div>
    <div class="paper-meta">
      📅 2024-03-17
      | 💬 10 pages, in processing of ACL 2024
    </div>
    <details class="paper-abstract">
      In recent years, the utilization of large language models for natural language dialogue has gained momentum, leading to their widespread adoption across various domains. However, their universal competence in addressing challenges specific to specialized fields such as law remains a subject of scrutiny. The incorporation of legal ethics into the model has been overlooked by researchers. We asserts that rigorous ethic evaluation is essential to ensure the effective integration of large language models in legal domains, emphasizing the need to assess domain-specific proficiency and domain-specific ethic. To address this, we propose a novelty evaluation methodology, utilizing authentic legal cases to evaluate the fundamental language abilities, specialized legal knowledge and legal robustness of large language models (LLMs). The findings from our comprehensive evaluation contribute significantly to the academic discourse surrounding the suitability and performance of large language models in legal domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.13063v2">Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-17
      | 💬 The paper is accepted by ICLR 2024. The code is publicly available at https://github.com/MiaoXiong2320/llm-uncertainty
    </div>
    <details class="paper-abstract">
      Empowering large language models to accurately express confidence in their answers is essential for trustworthy decision-making. Previous confidence elicitation methods, which primarily rely on white-box access to internal model information or model fine-tuning, have become less suitable for LLMs, especially closed-source commercial APIs. This leads to a growing need to explore the untapped area of black-box approaches for LLM uncertainty estimation. To better break down the problem, we define a systematic framework with three components: prompting strategies for eliciting verbalized confidence, sampling methods for generating multiple responses, and aggregation techniques for computing consistency. We then benchmark these methods on two key tasks-confidence calibration and failure prediction-across five types of datasets (e.g., commonsense and arithmetic reasoning) and five widely-used LLMs including GPT-4 and LLaMA 2 Chat. Our analysis uncovers several key insights: 1) LLMs, when verbalizing their confidence, tend to be overconfident, potentially imitating human patterns of expressing confidence. 2) As model capability scales up, both calibration and failure prediction performance improve. 3) Employing our proposed strategies, such as human-inspired prompts, consistency among multiple responses, and better aggregation strategies can help mitigate this overconfidence from various perspectives. 4) Comparisons with white-box methods indicate that while white-box methods perform better, the gap is narrow, e.g., 0.522 to 0.605 in AUROC. Despite these advancements, none of these techniques consistently outperform others, and all investigated methods struggle in challenging tasks, such as those requiring professional knowledge, indicating significant scope for improvement. We believe this study can serve as a strong baseline and provide insights for eliciting confidence in black-box LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01382v2">Compressing LLMs: The Truth is Rarely Pure and Never Simple</a></div>
    <div class="paper-meta">
      📅 2024-03-17
      | 💬 Accepted to ICLR 2024
    </div>
    <details class="paper-abstract">
      Despite their remarkable achievements, modern Large Language Models (LLMs) face exorbitant computational and memory footprints. Recently, several works have shown significant success in training-free and data-free compression (pruning and quantization) of LLMs that achieve 50 - 60% sparsity and reduce the bit width to 3 or 4 bits per weight, with negligible degradation of perplexity over the uncompressed baseline. As recent research efforts are focused on developing increasingly sophisticated compression methods, our work takes a step back and re-evaluates the effectiveness of existing SoTA compression methods, which rely on a fairly simple and widely questioned metric, perplexity (even for dense LLMs). We introduce Knowledge-Intensive Compressed LLM BenchmarK (LLM-KICK), a collection of carefully curated tasks to redefine the evaluation protocol for compressed LLMs, which have significant alignment with their dense counterparts and perplexity fail to capture subtle change in their true capabilities. LLM-KICK unveils many favorable merits and unfortunate plights of current SoTA compression methods: all pruning methods suffer significant performance degradation, sometimes at trivial sparsity ratios (e.g., 25-30%), and fail for N:M sparsity in knowledge-intensive tasks; current quantization methods are more successful than pruning; yet, pruned LLMs even at $\geq 50$% sparsity are robust in-context retrieval and summarization systems; among others. LLM-KICK is designed to holistically access compressed LLMs' ability for language understanding, reasoning, generation, in-context retrieval, in-context summarization, etc. We hope our study can foster the development of better LLM compression methods. The reproduced codes are available at https://github.com/VITA-Group/llm-kick.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13050v1">FlowMind: Automatic Workflow Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-17
      | 💬 Published in ACM ICAIF 2023
    </div>
    <details class="paper-abstract">
      The rapidly evolving field of Robotic Process Automation (RPA) has made significant strides in automating repetitive processes, yet its effectiveness diminishes in scenarios requiring spontaneous or unpredictable tasks demanded by users. This paper introduces a novel approach, FlowMind, leveraging the capabilities of Large Language Models (LLMs) such as Generative Pretrained Transformer (GPT), to address this limitation and create an automatic workflow generation system. In FlowMind, we propose a generic prompt recipe for a lecture that helps ground LLM reasoning with reliable Application Programming Interfaces (APIs). With this, FlowMind not only mitigates the common issue of hallucinations in LLMs, but also eliminates direct interaction between LLMs and proprietary data or code, thus ensuring the integrity and confidentiality of information - a cornerstone in financial services. FlowMind further simplifies user interaction by presenting high-level descriptions of auto-generated workflows, enabling users to inspect and provide feedback effectively. We also introduce NCEN-QA, a new dataset in finance for benchmarking question-answering tasks from N-CEN reports on funds. We used NCEN-QA to evaluate the performance of workflows generated by FlowMind against baseline and ablation variants of FlowMind. We demonstrate the success of FlowMind, the importance of each component in the proposed lecture recipe, and the effectiveness of user interaction and feedback in FlowMind.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10359v2">Can we Soft Prompt LLMs for Graph Learning Tasks?</a></div>
    <div class="paper-meta">
      📅 2024-03-16
      | 💬 Accepted by The Web Conference (WWW) 2024 Short Paper Track
    </div>
    <details class="paper-abstract">
      Graph plays an important role in representing complex relationships in real-world applications such as social networks, biological data and citation networks. In recent years, Large Language Models (LLMs) have achieved tremendous success in various domains, which makes applying LLMs to graphs particularly appealing. However, directly applying LLMs to graph modalities presents unique challenges due to the discrepancy and mismatch between the graph and text modalities. Hence, to further investigate LLMs' potential for comprehending graph information, we introduce GraphPrompter, a novel framework designed to align graph information with LLMs via soft prompts. Specifically, GraphPrompter consists of two main components: a graph neural network to encode complex graph information and an LLM that effectively processes textual information. Comprehensive experiments on various benchmark datasets under node classification and link prediction tasks demonstrate the effectiveness of our proposed method. The GraphPrompter framework unveils the substantial capabilities of LLMs as predictors in graph-related tasks, enabling researchers to utilize LLMs across a spectrum of real-world graph scenarios more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.01519v3">Exploring the Frontiers of LLMs in Psychological Applications: A Comprehensive Review</a></div>
    <div class="paper-meta">
      📅 2024-03-16
    </div>
    <details class="paper-abstract">
      This paper explores the frontiers of large language models (LLMs) in psychology applications. Psychology has undergone several theoretical changes, and the current use of Artificial Intelligence (AI) and Machine Learning, particularly LLMs, promises to open up new research directions. We provide a detailed exploration of how LLMs like ChatGPT are transforming psychological research. It discusses the impact of LLMs across various branches of psychology, including cognitive and behavioral, clinical and counseling, educational and developmental, and social and cultural psychology, highlighting their potential to simulate aspects of human cognition and behavior. The paper delves into the capabilities of these models to emulate human-like text generation, offering innovative tools for literature review, hypothesis generation, experimental design, experimental subjects, data analysis, academic writing, and peer review in psychology. While LLMs are essential in advancing research methodologies in psychology, the paper also cautions about their technical and ethical challenges. There are issues like data privacy, the ethical implications of using LLMs in psychological research, and the need for a deeper understanding of these models' limitations. Researchers should responsibly use LLMs in psychological studies, adhering to ethical standards and considering the potential consequences of deploying these technologies in sensitive areas. Overall, the article provides a comprehensive overview of the current state of LLMs in psychology, exploring potential benefits and challenges. It serves as a call to action for researchers to leverage LLMs' advantages responsibly while addressing associated risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01570v2">SERVAL: Synergy Learning between Vertical Models and LLMs towards Oracle-Level Zero-shot Medical Prediction</a></div>
    <div class="paper-meta">
      📅 2024-03-16
    </div>
    <details class="paper-abstract">
      Recent development of large language models (LLMs) has exhibited impressive zero-shot proficiency on generic and common sense questions. However, LLMs' application on domain-specific vertical questions still lags behind, primarily due to the humiliation problems and deficiencies in vertical knowledge. Furthermore, the vertical data annotation process often requires labor-intensive expert involvement, thereby presenting an additional challenge in enhancing the model's vertical capabilities. In this paper, we propose SERVAL, a synergy learning pipeline designed for unsupervised development of vertical capabilities in both LLMs and small models by mutual enhancement. Specifically, SERVAL utilizes the LLM's zero-shot outputs as annotations, leveraging its confidence to teach a robust vertical model from scratch. Reversely, the trained vertical model guides the LLM fine-tuning to enhance its zero-shot capability, progressively improving both models through an iterative process. In medical domain, known for complex vertical knowledge and costly annotations, comprehensive experiments show that, without access to any gold labels, SERVAL with the synergy learning of OpenAI GPT-3.5 and a simple model attains fully-supervised competitive performance across ten widely used medical datasets. These datasets represent vertically specialized medical diagnostic scenarios (e.g., diabetes, heart diseases, COVID-19), highlighting the potential of SERVAL in refining the vertical capabilities of LLMs and training vertical models from scratch, all achieved without the need for annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.07914v2">Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey</a></div>
    <div class="paper-meta">
      📅 2024-03-16
      | 💬 Accepted Paper in NAACL 2024
    </div>
    <details class="paper-abstract">
      The contemporary LLMs are prone to producing hallucinations, stemming mainly from the knowledge gaps within the models. To address this critical limitation, researchers employ diverse strategies to augment the LLMs by incorporating external knowledge, aiming to reduce hallucinations and enhance reasoning accuracy. Among these strategies, leveraging knowledge graphs as a source of external information has demonstrated promising results. In this survey, we comprehensively review these knowledge-graph-based augmentation techniques in LLMs, focusing on their efficacy in mitigating hallucinations. We systematically categorize these methods into three overarching groups, offering methodological comparisons and performance evaluations. Lastly, this survey explores the current trends and challenges associated with these techniques and outlines potential avenues for future research in this emerging field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10779v1">LLM-based Conversational AI Therapist for Daily Functioning Screening and Psychotherapeutic Intervention via Everyday Smart Devices</a></div>
    <div class="paper-meta">
      📅 2024-03-16
    </div>
    <details class="paper-abstract">
      Despite the global mental health crisis, access to screenings, professionals, and treatments remains high. In collaboration with licensed psychotherapists, we propose a Conversational AI Therapist with psychotherapeutic Interventions (CaiTI), a platform that leverages large language models (LLM)s and smart devices to enable better mental health self-care. CaiTI can screen the day-to-day functioning using natural and psychotherapeutic conversations. CaiTI leverages reinforcement learning to provide personalized conversation flow. CaiTI can accurately understand and interpret user responses. When the user needs further attention during the conversation, CaiTI can provide conversational psychotherapeutic interventions, including cognitive behavioral therapy (CBT) and motivational interviewing (MI). Leveraging the datasets prepared by the licensed psychotherapists, we experiment and microbenchmark various LLMs' performance in tasks along CaiTI's conversation flow and discuss their strengths and weaknesses. With the psychotherapists, we implement CaiTI and conduct 14-day and 24-week studies. The study results, validated by therapists, demonstrate that CaiTI can converse with users naturally, accurately understand and interpret user responses, and provide psychotherapeutic interventions appropriately and effectively. We showcase the potential of CaiTI LLMs to assist the mental therapy diagnosis and treatment and improve day-to-day functioning screening and precautionary psychotherapeutic intervention systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10507v1">Demystifying Faulty Code with LLM: Step-by-Step Reasoning for Explainable Fault Localization</a></div>
    <div class="paper-meta">
      📅 2024-03-15
      | 💬 To be appeared at 2024 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER)
    </div>
    <details class="paper-abstract">
      Fault localization is a critical process that involves identifying specific program elements responsible for program failures. Manually pinpointing these elements, such as classes, methods, or statements, which are associated with a fault is laborious and time-consuming. To overcome this challenge, various fault localization tools have been developed. These tools typically generate a ranked list of suspicious program elements. However, this information alone is insufficient. A prior study emphasized that automated fault localization should offer a rationale. In this study, we investigate the step-by-step reasoning for explainable fault localization. We explore the potential of Large Language Models (LLM) in assisting developers in reasoning about code. We proposed FuseFL that utilizes several combinations of information to enhance the LLM results which are spectrum-based fault localization results, test case execution outcomes, and code description (i.e., explanation of what the given code is intended to do). We conducted our investigation using faulty code from Refactory dataset. First, we evaluate the performance of the automated fault localization. Our results demonstrate a more than 30% increase in the number of successfully localized faults at Top-1 compared to the baseline. To evaluate the explanations generated by FuseFL, we create a dataset of human explanations that provide step-by-step reasoning as to why specific lines of code are considered faulty. This dataset consists of 324 faulty code files, along with explanations for 600 faulty lines. Furthermore, we also conducted human studies to evaluate the explanations. We found that for 22 out of the 30 randomly sampled cases, FuseFL generated correct explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10588v1">S3LLM: Large-Scale Scientific Software Understanding with LLMs using Source, Metadata, and Document</a></div>
    <div class="paper-meta">
      📅 2024-03-15
    </div>
    <details class="paper-abstract">
      The understanding of large-scale scientific software poses significant challenges due to its diverse codebase, extensive code length, and target computing architectures. The emergence of generative AI, specifically large language models (LLMs), provides novel pathways for understanding such complex scientific codes. This paper presents S3LLM, an LLM-based framework designed to enable the examination of source code, code metadata, and summarized information in conjunction with textual technical reports in an interactive, conversational manner through a user-friendly interface. S3LLM leverages open-source LLaMA-2 models to enhance code analysis through the automatic transformation of natural language queries into domain-specific language (DSL) queries. Specifically, it translates these queries into Feature Query Language (FQL), enabling efficient scanning and parsing of entire code repositories. In addition, S3LLM is equipped to handle diverse metadata types, including DOT, SQL, and customized formats. Furthermore, S3LLM incorporates retrieval augmented generation (RAG) and LangChain technologies to directly query extensive documents. S3LLM demonstrates the potential of using locally deployed open-source LLMs for the rapid understanding of large-scale scientific computing software, eliminating the need for extensive coding expertise, and thereby making the process more efficient and effective. S3LLM is available at https://github.com/ResponsibleAILab/s3llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10446v1">Enhancing LLM Factual Accuracy with RAG to Counter Hallucinations: A Case Study on Domain-Specific Queries in Private Knowledge-Bases</a></div>
    <div class="paper-meta">
      📅 2024-03-15
      | 💬 These authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      We proposed an end-to-end system design towards utilizing Retrieval Augmented Generation (RAG) to improve the factual accuracy of Large Language Models (LLMs) for domain-specific and time-sensitive queries related to private knowledge-bases. Our system integrates RAG pipeline with upstream datasets processing and downstream performance evaluation. Addressing the challenge of LLM hallucinations, we finetune models with a curated dataset which originates from CMU's extensive resources and annotated with the teacher model. Our experiments demonstrate the system's effectiveness in generating more accurate answers to domain-specific and time-sensitive inquiries. The results also revealed the limitations of fine-tuning LLMs with small-scale and skewed datasets. This research highlights the potential of RAG systems in augmenting LLMs with external datasets for improved performance in knowledge-intensive tasks. Our code and models are available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10407v1">A Thorough Comparison of Cross-Encoders and LLMs for Reranking SPLADE</a></div>
    <div class="paper-meta">
      📅 2024-03-15
    </div>
    <details class="paper-abstract">
      We present a comparative study between cross-encoder and LLMs rerankers in the context of re-ranking effective SPLADE retrievers. We conduct a large evaluation on TREC Deep Learning datasets and out-of-domain datasets such as BEIR and LoTTE. In the first set of experiments, we show how cross-encoder rerankers are hard to distinguish when it comes to re-rerank SPLADE on MS MARCO. Observations shift in the out-of-domain scenario, where both the type of model and the number of documents to re-rank have an impact on effectiveness. Then, we focus on listwise rerankers based on Large Language Models -- especially GPT-4. While GPT-4 demonstrates impressive (zero-shot) performance, we show that traditional cross-encoders remain very competitive. Overall, our findings aim to to provide a more nuanced perspective on the recent excitement surrounding LLM-based re-rankers -- by positioning them as another factor to consider in balancing effectiveness and efficiency in search systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10228v1">HawkEye: Training Video-Text LLMs for Grounding Text in Videos</a></div>
    <div class="paper-meta">
      📅 2024-03-15
    </div>
    <details class="paper-abstract">
      Video-text Large Language Models (video-text LLMs) have shown remarkable performance in answering questions and holding conversations on simple videos. However, they perform almost the same as random on grounding text queries in long and complicated videos, having little ability to understand and reason about temporal information, which is the most fundamental difference between videos and images. In this paper, we propose HawkEye, one of the first video-text LLMs that can perform temporal video grounding in a fully text-to-text manner. To collect training data that is applicable for temporal video grounding, we construct InternVid-G, a large-scale video-text corpus with segment-level captions and negative spans, with which we introduce two new time-aware training objectives to video-text LLMs. We also propose a coarse-grained method of representing segments in videos, which is more robust and easier for LLMs to learn and follow than other alternatives. Extensive experiments show that HawkEye is better at temporal video grounding and comparable on other video-text tasks with existing video-text LLMs, which verifies its superior video-text multi-modal understanding abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10082v1">CrossGLG: LLM Guides One-shot Skeleton-based 3D Action Recognition in a Cross-level Manner</a></div>
    <div class="paper-meta">
      📅 2024-03-15
    </div>
    <details class="paper-abstract">
      Most existing one-shot skeleton-based action recognition focuses on raw low-level information (e.g., joint location), and may suffer from local information loss and low generalization ability. To alleviate these, we propose to leverage text description generated from large language models (LLM) that contain high-level human knowledge, to guide feature learning, in a global-local-global way. Particularly, during training, we design $2$ prompts to gain global and local text descriptions of each action from an LLM. We first utilize the global text description to guide the skeleton encoder focus on informative joints (i.e.,global-to-local). Then we build non-local interaction between local text and joint features, to form the final global representation (i.e., local-to-global). To mitigate the asymmetry issue between the training and inference phases, we further design a dual-branch architecture that allows the model to perform novel class inference without any text input, also making the additional inference cost neglectable compared with the base skeleton encoder. Extensive experiments on three different benchmarks show that CrossGLG consistently outperforms the existing SOTA methods with large margins, and the inference cost (model size) is only $2.8$\% than the previous SOTA. CrossGLG can also serve as a plug-and-play module that can substantially enhance the performance of different SOTA skeleton encoders with a neglectable cost during inference. The source code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07947v1">ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-03-15
      | 💬 Accepted to ASPLOS 2024 (summer cycle)
    </div>
    <details class="paper-abstract">
      This paper presents ExeGPT, a distributed system designed for constraint-aware LLM inference. ExeGPT finds and runs with an optimal execution schedule to maximize inference throughput while satisfying a given latency constraint. By leveraging the distribution of input and output sequences, it effectively allocates resources and determines optimal execution configurations, including batch sizes and partial tensor parallelism. We also introduce two scheduling strategies based on Round-Robin Allocation and Workload-Aware Allocation policies, suitable for different NLP workloads. We evaluate ExeGPT on six LLM instances of T5, OPT, and GPT-3 and five NLP tasks, each with four distinct latency constraints. Compared to FasterTransformer, ExeGPT achieves up to 15.2x improvements in throughput and 6x improvements in latency. Overall, ExeGPT achieves an average throughput gain of 2.9x across twenty evaluation scenarios. Moreover, when adapting to changing sequence distributions, the cost of adjusting the schedule in ExeGPT is reasonably modest. ExeGPT proves to be an effective solution for optimizing and executing LLM inference for diverse NLP workload and serving conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.07317v2">Platypus: Quick, Cheap, and Powerful Refinement of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-14
      | 💬 Workshop on Instruction Tuning and Instruction Following at NeurIPS 2023
    </div>
    <details class="paper-abstract">
      We present $\textbf{Platypus}$, a family of fine-tuned and merged Large Language Models (LLMs) that achieves the strongest performance and currently stands at first place in HuggingFace's Open LLM Leaderboard as of the release date of this work. In this work we describe (1) our curated dataset $\textbf{Open-Platypus}$, that is a subset of other open datasets and which $\textit{we release to the public}$ (2) our process of fine-tuning and merging LoRA modules in order to conserve the strong prior of pretrained LLMs, while bringing specific domain knowledge to the surface (3) our efforts in checking for test data leaks and contamination in the training data, which can inform future research. Specifically, the Platypus family achieves strong performance in quantitative LLM metrics across model sizes, topping the global Open LLM leaderboard while using just a fraction of the fine-tuning data and overall compute that are required for other state-of-the-art fine-tuned LLMs. In particular, a 13B Platypus model can be trained on $\textit{a single}$ A100 GPU using 25k questions in 5 hours. This is a testament of the quality of our Open-Platypus dataset, and opens opportunities for more improvements in the field. Project page: https://platypus-llm.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01308v2">VBART: The Turkish LLM</a></div>
    <div class="paper-meta">
      📅 2024-03-14
    </div>
    <details class="paper-abstract">
      We present VBART, the first Turkish sequence-to-sequence Large Language Models (LLMs) pre-trained on a large corpus from scratch. VBART are compact LLMs based on good ideas leveraged from BART and mBART models and come in two sizes, Large and XLarge. Fine-tuned VBART models surpass the prior state-of-the-art results in abstractive text summarization, title generation, text paraphrasing, question answering and question generation tasks. They allow fine-tuning for future text generation tasks and datasets, carving a new path for Turkish Natural Language Processing (NLP) research. Our work shows that having a pre-trained LLM for Turkish outperforms up to 3x multilingual models, improving existing results and providing efficient models for training and inference. Moreover, we show that our monolingual tokenizer is up to 11x more efficient than multilingual tokenizers. Last but not least, we introduce a method to enlarge an existing pre-trained LLM and question the relevancy of Chinchilla Scaling Law to sequence-to-sequence masked language models. Our fine-tuned models, tokenizer and cleaned vngrs-web-corpus of 135 GB are publicly available at huggingface.co/vngrs-ai.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09537v1">Analyzing and Mitigating (with LLMs) the Security Misconfigurations of Helm Charts from Artifact Hub</a></div>
    <div class="paper-meta">
      📅 2024-03-14
      | 💬 MSR 2024 - Registered Reports
    </div>
    <details class="paper-abstract">
      Background: Helm is a package manager that allows defining, installing, and upgrading applications with Kubernetes (K8s), a popular container orchestration platform. A Helm chart is a collection of files describing all dependencies, resources, and parameters required for deploying an application within a K8s cluster. Objective: The goal of this study is to mine and empirically evaluate the security of Helm charts, comparing the performance of existing tools in terms of misconfigurations reported by policies available by default, and measure to what extent LLMs could be used for removing misconfiguration. We also want to investigate whether there are false positives in both the LLM refactorings and the tool outputs. Method: We propose a pipeline to mine Helm charts from Artifact Hub, a popular centralized repository, and analyze them using state-of-the-art open-source tools, such as Checkov and KICS. First, such a pipeline will run several chart analyzers and identify the common and unique misconfigurations reported by each tool. Secondly, it will use LLMs to suggest mitigation for each misconfiguration. Finally, the chart refactoring previously generated will be analyzed again by the same tools to see whether it satisfies the tool's policies. At the same time, we will also perform a manual analysis on a subset of charts to evaluate whether there are false positive misconfigurations from the tool's reporting and in the LLM refactoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09442v1">LLM-based agents for automating the enhancement of user story quality: An early report</a></div>
    <div class="paper-meta">
      📅 2024-03-14
      | 💬 16 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      In agile software development, maintaining high-quality user stories is crucial, but also challenging. This study explores the use of large language models to automatically improve the user story quality in Austrian Post Group IT agile teams. We developed a reference model for an Autonomous LLM-based Agent System and implemented it at the company. The quality of user stories in the study and the effectiveness of these agents for user story quality improvement was assessed by 11 participants across six agile teams. Our findings demonstrate the potential of LLMs in improving user story quality, contributing to the research on AI role in agile development, and providing a practical example of the transformative impact of AI in an industry setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05156v2">On Protecting the Data Privacy of Large Language Models (LLMs): A Survey</a></div>
    <div class="paper-meta">
      📅 2024-03-14
      | 💬 18 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are complex artificial intelligence systems capable of understanding, generating and translating human language. They learn language patterns by analyzing large amounts of text data, allowing them to perform writing, conversation, summarizing and other language tasks. When LLMs process and generate large amounts of data, there is a risk of leaking sensitive information, which may threaten data privacy. This paper concentrates on elucidating the data privacy concerns associated with LLMs to foster a comprehensive understanding. Specifically, a thorough investigation is undertaken to delineate the spectrum of data privacy threats, encompassing both passive privacy leakage and active privacy attacks within LLMs. Subsequently, we conduct an assessment of the privacy protection mechanisms employed by LLMs at various stages, followed by a detailed examination of their efficacy and constraints. Finally, the discourse extends to delineate the challenges encountered and outline prospective directions for advancement in the realm of LLM privacy protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09148v1">Evaluating LLMs for Gender Disparities in Notable Persons</a></div>
    <div class="paper-meta">
      📅 2024-03-14
    </div>
    <details class="paper-abstract">
      This study examines the use of Large Language Models (LLMs) for retrieving factual information, addressing concerns over their propensity to produce factually incorrect "hallucinated" responses or to altogether decline to even answer prompt at all. Specifically, it investigates the presence of gender-based biases in LLMs' responses to factual inquiries. This paper takes a multi-pronged approach to evaluating GPT models by evaluating fairness across multiple dimensions of recall, hallucinations and declinations. Our findings reveal discernible gender disparities in the responses generated by GPT-3.5. While advancements in GPT-4 have led to improvements in performance, they have not fully eradicated these gender disparities, notably in instances where responses are declined. The study further explores the origins of these disparities by examining the influence of gender associations in prompts and the homogeneity in the responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07969v2">KnowCoder: Coding Structured Knowledge into LLMs for Universal Information Extraction</a></div>
    <div class="paper-meta">
      📅 2024-03-14
    </div>
    <details class="paper-abstract">
      In this paper, we propose KnowCoder, a Large Language Model (LLM) to conduct Universal Information Extraction (UIE) via code generation. KnowCoder aims to develop a kind of unified schema representation that LLMs can easily understand and an effective learning framework that encourages LLMs to follow schemas and extract structured knowledge accurately. To achieve these, KnowCoder introduces a code-style schema representation method to uniformly transform different schemas into Python classes, with which complex schema information, such as constraints among tasks in UIE, can be captured in an LLM-friendly manner. We further construct a code-style schema library covering over $\textbf{30,000}$ types of knowledge, which is the largest one for UIE, to the best of our knowledge. To ease the learning process of LLMs, KnowCoder contains a two-phase learning framework that enhances its schema understanding ability via code pretraining and its schema following ability via instruction tuning. After code pretraining on around $1.5$B automatically constructed data, KnowCoder already attains remarkable generalization ability and achieves relative improvements by $\textbf{49.8%}$ F1, compared to LLaMA2, under the few-shot setting. After instruction tuning, KnowCoder further exhibits strong generalization ability on unseen schemas and achieves up to $\textbf{12.5%}$ and $\textbf{21.9%}$, compared to sota baselines, under the zero-shot setting and the low resource setting, respectively. Additionally, based on our unified schema representations, various human-annotated datasets can simultaneously be utilized to refine KnowCoder, which achieves significant improvements up to $\textbf{7.5%}$ under the supervised setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08946v1">Usable XAI: 10 Strategies Towards Exploiting Explainability in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2024-03-13
      | 💬 38 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Explainable AI (XAI) refers to techniques that provide human-understandable insights into the workings of AI models. Recently, the focus of XAI is being extended towards Large Language Models (LLMs) which are often criticized for their lack of transparency. This extension calls for a significant transformation in XAI methodologies because of two reasons. First, many existing XAI methods cannot be directly applied to LLMs due to their complexity advanced capabilities. Second, as LLMs are increasingly deployed across diverse industry applications, the role of XAI shifts from merely opening the "black box" to actively enhancing the productivity and applicability of LLMs in real-world settings. Meanwhile, unlike traditional machine learning models that are passive recipients of XAI insights, the distinct abilities of LLMs can reciprocally enhance XAI. Therefore, in this paper, we introduce Usable XAI in the context of LLMs by analyzing (1) how XAI can benefit LLMs and AI systems, and (2) how LLMs can contribute to the advancement of XAI. We introduce 10 strategies, introducing the key techniques for each and discussing their associated challenges. We also provide case studies to demonstrate how to obtain and leverage explanations. The code used in this paper can be found at: https://github.com/JacksonWuxs/UsableXAI_LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09740v1">Teaching Machines to Code: Smart Contract Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-13
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has marked a significant milestone in the realm of artificial intelligence, with their capabilities often matching or surpassing human expertise in various domains. Among these achievements, their adeptness in translation tasks stands out, closely mimicking the intricate and preliminary processes undertaken by human translators to ensure the fidelity and quality of the translated content. Despite the advancements in utilizing LLMs for translating programming code across different languages, the domain of smart contract translation, particularly into languages not previously encountered by the LLM, remains largely unexplored. In our research, we present a pioneering approach, SolMover, which harnesses the synergy of two distinct LLMs within a unified framework. This framework is designed to grasp coding principles and apply this understanding to the translation of code into an unfamiliar language. Our study delves into the capacity of LLMs to mimic human learning processes, offering an in-depth evaluation of our methodology for converting smart contracts written in Solidity to Move, a language with limited resources. The framework employs one LLM to decipher coding conventions for the new language, creating a blueprint for the second LLM, which, lacking planning abilities, possesses coding expertise. The empirical evidence from our experiments suggests that SolMover substantially enhances performance compared to gpt-3.5-turbo-1106, and achieves superior results over competitors such as Palm2 and Mixtral-8x7B-Instruct. Additionally, our analysis highlights the efficacy of our bug mitigation strategy in elevating code quality across all models, even outside the SolMover framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08743v1">Steering LLMs Towards Unbiased Responses: A Causality-Guided Debiasing Framework</a></div>
    <div class="paper-meta">
      📅 2024-03-13
      | 💬 18 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can easily generate biased and discriminative responses. As LLMs tap into consequential decision-making (e.g., hiring and healthcare), it is of crucial importance to develop strategies to mitigate these biases. This paper focuses on social bias, tackling the association between demographic information and LLM outputs. We propose a causality-guided debiasing framework that utilizes causal understandings of (1) the data-generating process of the training corpus fed to LLMs, and (2) the internal reasoning process of LLM inference, to guide the design of prompts for debiasing LLM outputs through selection mechanisms. Our framework unifies existing de-biasing prompting approaches such as inhibitive instructions and in-context contrastive examples, and sheds light on new ways of debiasing by encouraging bias-free reasoning. Our strong empirical performance on real-world datasets demonstrates that our framework provides principled guidelines on debiasing LLM outputs even with only the black-box access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.00257v2">AMSP: Reducing Communication Overhead of ZeRO for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-03-13
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) encounters challenges in GPU memory consumption due to the high memory requirements of model states. The widely used Zero Redundancy Optimizer (ZeRO) addresses this issue through strategic sharding but introduces communication challenges at scale. To tackle this problem, we propose AMSP, a system designed to optimize ZeRO for scalable LLM training. AMSP incorporates three flexible sharding strategies: Full-Replica, Full-Sharding, and Partial-Sharding, and allows each component within the model states (Parameters, Gradients, Optimizer States) to independently choose a sharding strategy as well as the device mesh. We conduct a thorough analysis of communication costs, formulating an optimization problem to discover the optimal sharding strategy. Additionally, AMSP optimizes distributed LLM training by efficiently overlapping communication with computation. Evaluations demonstrate up to 52\% Model FLOPs Utilization (MFU) when training the LLaMA-based model on 1024 GPUs, resulting in a 1.56 times improvement in training throughput compared to newly proposed systems like MiCS and ZeRO++.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08430v1">Search-based Optimisation of LLM Learning Shots for Story Point Estimation</a></div>
    <div class="paper-meta">
      📅 2024-03-13
      | 💬 6 pages, Accepted at SSBSE'23 NIER Track
    </div>
    <details class="paper-abstract">
      One of the ways Large Language Models (LLMs) are used to perform machine learning tasks is to provide them with a few examples before asking them to produce a prediction. This is a meta-learning process known as few-shot learning. In this paper, we use available Search-Based methods to optimise the number and combination of examples that can improve an LLM's estimation performance, when it is used to estimate story points for new agile tasks. Our preliminary results show that our SBSE technique improves the estimation performance of the LLM by 59.34% on average (in terms of mean absolute error of the estimation) over three datasets against a zero-shot setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08429v1">Software Vulnerability and Functionality Assessment using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-13
      | 💬 4 pages, accepted to NLBSE'24
    </div>
    <details class="paper-abstract">
      While code review is central to the software development process, it can be tedious and expensive to carry out. In this paper, we investigate whether and how Large Language Models (LLMs) can aid with code reviews. Our investigation focuses on two tasks that we argue are fundamental to good reviews: (i) flagging code with security vulnerabilities and (ii) performing software functionality validation, i.e., ensuring that code meets its intended functionality. To test performance on both tasks, we use zero-shot and chain-of-thought prompting to obtain final ``approve or reject'' recommendations. As data, we employ seminal code generation datasets (HumanEval and MBPP) along with expert-written code snippets with security vulnerabilities from the Common Weakness Enumeration (CWE). Our experiments consider a mixture of three proprietary models from OpenAI and smaller open-source LLMs. We find that the former outperforms the latter by a large margin. Motivated by promising results, we finally ask our models to provide detailed descriptions of security vulnerabilities. Results show that 36.7% of LLM-generated descriptions can be associated with true CWE vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08396v1">A Picture Is Worth a Thousand Words: Exploring Diagram and Video-Based OOP Exercises to Counter LLM Over-Reliance</a></div>
    <div class="paper-meta">
      📅 2024-03-13
      | 💬 This is the author's draft of this paper
    </div>
    <details class="paper-abstract">
      Much research has highlighted the impressive capabilities of large language models (LLMs), like GPT and Bard, for solving introductory programming exercises. Recent work has shown that LLMs can effectively solve a range of more complex object-oriented programming (OOP) exercises with text-based specifications. This raises concerns about academic integrity, as students might use these models to complete assignments unethically, neglecting the development of important skills such as program design, problem-solving, and computational thinking. To address this, we propose an innovative approach to formulating OOP tasks using diagrams and videos, as a way to foster problem-solving and deter students from a copy-and-prompt approach in OOP courses. We introduce a novel notation system for specifying OOP assignments, encompassing structural and behavioral requirements, and assess its use in a classroom setting over a semester. Student perceptions of this approach are explored through a survey (n=56). Generally, students responded positively to diagrams and videos, with video-based projects being better received than diagram-based exercises. This notation appears to have several benefits, with students investing more effort in understanding the diagrams and feeling more motivated to engage with the video-based projects. Furthermore, students reported being less inclined to rely on LLM-based code generation tools for these diagram and video-based exercises. Experiments with GPT-4 and Bard's vision abilities revealed that they currently fall short in interpreting these diagrams to generate accurate code solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08345v1">From human experts to machines: An LLM supported approach to ontology and knowledge graph construction</a></div>
    <div class="paper-meta">
      📅 2024-03-13
    </div>
    <details class="paper-abstract">
      The conventional process of building Ontologies and Knowledge Graphs (KGs) heavily relies on human domain experts to define entities and relationship types, establish hierarchies, maintain relevance to the domain, fill the ABox (or populate with instances), and ensure data quality (including amongst others accuracy and completeness). On the other hand, Large Language Models (LLMs) have recently gained popularity for their ability to understand and generate human-like natural language, offering promising ways to automate aspects of this process. This work explores the (semi-)automatic construction of KGs facilitated by open-source LLMs. Our pipeline involves formulating competency questions (CQs), developing an ontology (TBox) based on these CQs, constructing KGs using the developed ontology, and evaluating the resultant KG with minimal to no involvement of human experts. We showcase the feasibility of our semi-automated pipeline by creating a KG on deep learning methodologies by exploiting scholarly publications. To evaluate the answers generated via Retrieval-Augmented-Generation (RAG) as well as the KG concepts automatically extracted using LLMs, we design a judge LLM, which rates the generated content based on ground truth. Our findings suggest that employing LLMs could potentially reduce the human effort involved in the construction of KGs, although a human-in-the-loop approach is recommended to evaluate automatically generated KGs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.17428v2">CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets</a></div>
    <div class="paper-meta">
      📅 2024-03-13
      | 💬 Accepted to ICLR 2024. Code is available at https://github.com/lifan-yuan/CRAFT
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often augmented with tools to solve complex tasks. By generating code snippets and executing them through task-specific Application Programming Interfaces (APIs), they can offload certain functions to dedicated external modules, such as image encoding and performing calculations. However, most existing approaches to augment LLMs with tools are constrained by general-purpose APIs and lack the flexibility for tailoring them to specific tasks. In this work, we present CRAFT, a general tool creation and retrieval framework for LLMs. It creates toolsets specifically curated for the tasks and equips LLMs with a component that retrieves tools from these sets to enhance their capability to solve complex tasks. For each task, we collect specific code solutions by prompting GPT-4 to solve the training examples. Following a validation step ensuring the correctness, these solutions are abstracted into code snippets to enhance reusability, and deduplicated for higher quality. At inference time, the language model retrieves snippets from the toolsets and then executes them or generates the output conditioning on the retrieved snippets. Our method is designed to be flexible and offers a plug-and-play approach to adapt off-the-shelf LLMs to unseen domains and modalities, without any finetuning. Experiments on vision-language, tabular processing, and mathematical reasoning tasks show that our approach achieves substantial improvements compared to strong baselines. In addition, our in-depth analysis reveals that: (1) consistent performance improvement can be achieved by scaling up the number of tools and the capability of the backbone models; (2) each component of our approach contributes to the performance gains; (3) the created tools are well-structured and reliable with low complexity and atomicity. The code is available at https://github.com/lifan-yuan/CRAFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10553v1">Learning to Watermark LLM-generated Text via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-03-13
    </div>
    <details class="paper-abstract">
      We study how to watermark LLM outputs, i.e. embedding algorithmically detectable signals into LLM-generated text to track misuse. Unlike the current mainstream methods that work with a fixed LLM, we expand the watermark design space by including the LLM tuning stage in the watermark pipeline. While prior works focus on token-level watermark that embeds signals into the output, we design a model-level watermark that embeds signals into the LLM weights, and such signals can be detected by a paired detector. We propose a co-training framework based on reinforcement learning that iteratively (1) trains a detector to detect the generated watermarked text and (2) tunes the LLM to generate text easily detectable by the detector while keeping its normal utility. We empirically show that our watermarks are more accurate, robust, and adaptable (to new attacks). It also allows watermarked model open-sourcing. In addition, if used together with alignment, the extra overhead introduced is low - only training an extra reward model (i.e. our detector). We hope our work can bring more effort into studying a broader watermark design that is not limited to working with a fixed LLM. We open-source the code: https://github.com/xiaojunxu/learning-to-watermark-llm .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2211.00635v3">Two-stage LLM Fine-tuning with Less Specialization and More Generalization</a></div>
    <div class="paper-meta">
      📅 2024-03-12
      | 💬 ICLR 2024
    </div>
    <details class="paper-abstract">
      Pretrained large language models (LLMs) are general purpose problem solvers applicable to a diverse set of tasks with prompts. They can be further improved towards a specific task by fine-tuning on a specialized dataset. However, fine-tuning usually makes the model narrowly specialized on this dataset with reduced general in-context learning performances, which is undesirable whenever the fine-tuned model needs to handle additional tasks where no fine-tuning data is available. In this work, we first demonstrate that fine-tuning on a single task indeed decreases LLMs' general in-context learning performance. We discover one important cause of such forgetting, format specialization, where the model overfits to the format of the fine-tuned task.We further show that format specialization happens at the very beginning of fine-tuning. To solve this problem, we propose Prompt Tuning with MOdel Tuning (ProMoT), a simple yet effective two-stage fine-tuning framework that reduces format specialization and improves generalization.ProMoT offloads task-specific format learning into additional and removable parameters by first doing prompt tuning and then fine-tuning the model itself with this soft prompt attached. With experiments on several fine-tuning tasks and 8 in-context evaluation tasks, we show that ProMoT achieves comparable performance on fine-tuned tasks to standard fine-tuning, but with much less loss of in-context learning performances across a board range of out-of-domain evaluation tasks. More importantly, ProMoT can even enhance generalization on in-context learning tasks that are semantically related to the fine-tuned task, e.g. ProMoT on En-Fr translation significantly improves performance on other language pairs, and ProMoT on NLI improves performance on summarization. Experiments also show that ProMoT can improve the generalization performance of multi-task training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08032v1">LG-Traj: LLM Guided Pedestrian Trajectory Prediction</a></div>
    <div class="paper-meta">
      📅 2024-03-12
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Accurate pedestrian trajectory prediction is crucial for various applications, and it requires a deep understanding of pedestrian motion patterns in dynamic environments. However, existing pedestrian trajectory prediction methods still need more exploration to fully leverage these motion patterns. This paper investigates the possibilities of using Large Language Models (LLMs) to improve pedestrian trajectory prediction tasks by inducing motion cues. We introduce LG-Traj, a novel approach incorporating LLMs to generate motion cues present in pedestrian past/observed trajectories. Our approach also incorporates motion cues present in pedestrian future trajectories by clustering future trajectories of training data using a mixture of Gaussians. These motion cues, along with pedestrian coordinates, facilitate a better understanding of the underlying representation. Furthermore, we utilize singular value decomposition to augment the observed trajectories, incorporating them into the model learning process to further enhance representation learning. Our method employs a transformer-based architecture comprising a motion encoder to model motion patterns and a social decoder to capture social interactions among pedestrians. We demonstrate the effectiveness of our approach on popular pedestrian trajectory prediction benchmarks, namely ETH-UCY and SDD, and present various ablation experiments to validate our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07816v1">Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM</a></div>
    <div class="paper-meta">
      📅 2024-03-12
    </div>
    <details class="paper-abstract">
      We investigate efficient methods for training Large Language Models (LLMs) to possess capabilities in multiple specialized domains, such as coding, math reasoning and world knowledge. Our method, named Branch-Train-MiX (BTX), starts from a seed model, which is branched to train experts in embarrassingly parallel fashion with high throughput and reduced communication cost. After individual experts are asynchronously trained, BTX brings together their feedforward parameters as experts in Mixture-of-Expert (MoE) layers and averages the remaining parameters, followed by an MoE-finetuning stage to learn token-level routing. BTX generalizes two special cases, the Branch-Train-Merge method, which does not have the MoE finetuning stage to learn routing, and sparse upcycling, which omits the stage of training experts asynchronously. Compared to alternative approaches, BTX achieves the best accuracy-efficiency tradeoff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.10691v3">MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback</a></div>
    <div class="paper-meta">
      📅 2024-03-12
      | 💬 ICLR 2024. Code is available on our project website: https://xingyaoww.github.io/mint-bench
    </div>
    <details class="paper-abstract">
      To solve complex tasks, large language models (LLMs) often require multiple rounds of interactions with the user, sometimes assisted by external tools. However, current evaluation protocols often emphasize benchmark performance with single-turn exchanges, neglecting the nuanced interactions among the user, LLMs, and external tools, while also underestimating the importance of natural language feedback from users. These oversights contribute to discrepancies between research benchmark evaluations and real-world use cases. We introduce MINT, a benchmark that evaluates LLMs' ability to solve tasks with multi-turn interactions by (1) using tools and (2) leveraging natural language feedback. To ensure reproducibility, we provide an evaluation framework where LLMs can access tools by executing Python code and receive users' natural language feedback simulated by GPT-4. We repurpose a diverse set of established evaluation datasets focusing on reasoning, coding, and decision-making and carefully curate them into a compact subset for efficient evaluation. Our analysis of 20 open- and closed-source LLMs offers intriguing findings. (a) LLMs generally benefit from tools and language feedback, with performance gains (absolute, same below) of 1-8% for each turn of tool use and 2-17% with natural language feedback. (b) Better single-turn performance does not guarantee better multi-turn performance. (c) Surprisingly, on the LLMs evaluated, supervised instruction-finetuning (SIFT) and reinforcement learning from human feedback (RLHF) generally hurt multi-turn capabilities. We expect MINT can help measure progress and incentivize research in improving LLMs' capabilities in multi-turn interactions, especially for open-source communities where multi-turn human evaluation can be less accessible compared to commercial LLMs with a larger user base.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07557v1">SIFiD: Reassess Summary Factual Inconsistency Detection with LLM</a></div>
    <div class="paper-meta">
      📅 2024-03-12
    </div>
    <details class="paper-abstract">
      Ensuring factual consistency between the summary and the original document is paramount in summarization tasks. Consequently, considerable effort has been dedicated to detecting inconsistencies. With the advent of Large Language Models (LLMs), recent studies have begun to leverage their advanced language understanding capabilities for inconsistency detection. However, early attempts have shown that LLMs underperform traditional models due to their limited ability to follow instructions and the absence of an effective detection methodology. In this study, we reassess summary inconsistency detection with LLMs, comparing the performances of GPT-3.5 and GPT-4. To advance research in LLM-based inconsistency detection, we propose SIFiD (Summary Inconsistency Detection with Filtered Document) that identify key sentences within documents by either employing natural language inference or measuring semantic similarity between summaries and documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15459v3">Multi-LLM Collaboration + Data-Centric Innovation = 2x Better Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2024-03-12
      | 💬 Accepted in the ICSE 2024 Research Track with a different title "Out of Sight, Out of Mind: Better Automatic Vulnerability Repair by Broadening Input Ranges and Sources"
    </div>
    <details class="paper-abstract">
      The advances of deep learning (DL) have paved the way for automatic software vulnerability repair approaches, which effectively learn the mapping from the vulnerable code to the fixed code. Nevertheless, existing DL-based vulnerability repair methods face notable limitations: 1) they struggle to handle lengthy vulnerable code, 2) they treat code as natural language texts, neglecting its inherent structure, and 3) they do not tap into the valuable expert knowledge present in the expert system. To address this, we propose VulMaster, a Transformer-based neural network model that excels at generating vulnerability repairs through data-centric innovation. Specifically, VulMaster introduces the utilization and combination of various types of input data, including complete vulnerable code of any size, vulnerable code structures, and expert knowledge from the CWE system. Additionally, VulMaster leverages the collaboration between two Large Language Models (LLMs), CodeT5 and ChatGPT: CodeT5 acts as the customizable backbone LLM, fine-tuned with the training data, while ChatGPT supplements by providing missing relevant inputs to CodeT5. We evaluated VulMaster on a real-world C/C++ vulnerability repair dataset comprising 1,754 projects with 5,800 vulnerable functions. The experimental results demonstrated that VulMaster exhibits substantial improvements compared to the learning-based state-of-the-art vulnerability repair approach. Specifically, VulMaster improves the EM, BLEU, and CodeBLEU scores from 10.2\% to 20.0\%, 21.3\% to 29.3\%, and 32.5\% to 40.9\%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07376v1">NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-03-12
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN), as a crucial research problem of Embodied AI, requires an embodied agent to navigate through complex 3D environments following natural language instructions. Recent research has highlighted the promising capacity of large language models (LLMs) in VLN by improving navigational reasoning accuracy and interpretability. However, their predominant use in an offline manner usually suffers from substantial domain gap between the VLN task and the LLM training corpus. This paper introduces a novel strategy called Navigational Chain-of-Thought (NavCoT), where we fulfill parameter-efficient in-domain training to enable self-guided navigational decision, leading to a significant mitigation of the domain gap in a cost-effective manner. Specifically, at each timestep, the LLM is prompted to forecast the navigational chain-of-thought by: 1) acting as a world model to imagine the next observation according to the instruction, 2) selecting the candidate observation that best aligns with the imagination, and 3) determining the action based on the reasoning from the prior steps. Through constructing formalized labels for training, the LLM can learn to generate desired and reasonable chain-of-thought outputs for improving the action decision. Experimental results across various training settings and popular VLN benchmarks (e.g., Room-to-Room (R2R), Room-across-Room (RxR), Room-for-Room (R4R)) show the significant superiority of NavCoT over the direct action prediction variants. Through simple parameter-efficient finetuning, our NavCoT outperforms a recent GPT4-based approach with ~7% relative improvement on the R2R dataset. We believe that NavCoT will help unlock more task-adaptive and scalable LLM-based embodied agents, which are helpful for developing real-world robotics applications. Code is available at https://github.com/expectorlin/NavCoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07283v1">A Framework for Cost-Effective and Self-Adaptive LLM Shaking and Recovery Mechanism</a></div>
    <div class="paper-meta">
      📅 2024-03-12
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) gain great success in real-world applications, an increasing number of users are seeking to develop and deploy their customized LLMs through cloud services. Nonetheless, in some specific domains, there are still concerns regarding cost and trade-offs between privacy issues and accuracy. In this study, we introduce a cost-effective and self-adaptive LLM shaking tuning and recovery mechanism, named CypherTalk. With carefully designed horizontal and vertical shaking operators, we can achieve comparable accuracy results with SOTA privacy-preserving LLM schemes using Cryptography-based or Differential Privacy-based methods. Experiments also show that with the CypherTalk framework, users can achieve reliable accuracy when using optimized shaking operator settings. To our best knowledge, this is the first work that considers cost, and trade-off between model utility and privacy in LLM scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05538v2">Explaining Code Examples in Introductory Programming Courses: LLM vs Humans</a></div>
    <div class="paper-meta">
      📅 2024-03-12
      | 💬 3 tables; 1 figure
    </div>
    <details class="paper-abstract">
      Worked examples, which present an explained code for solving typical programming problems are among the most popular types of learning content in programming classes. Most approaches and tools for presenting these examples to students are based on line-by-line explanations of the example code. However, instructors rarely have time to provide explanations for many examples typically used in a programming class. In this paper, we assess the feasibility of using LLMs to generate code explanations for passive and active example exploration systems. To achieve this goal, we compare the code explanations generated by chatGPT with the explanations generated by both experts and students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07213v1">Which LLM to Play? Convergence-Aware Online Model Selection with Time-Increasing Bandits</a></div>
    <div class="paper-meta">
      📅 2024-03-11
      | 💬 Accepted by WWW'24 (Oral)
    </div>
    <details class="paper-abstract">
      Web-based applications such as chatbots, search engines and news recommendations continue to grow in scale and complexity with the recent surge in the adoption of LLMs. Online model selection has thus garnered increasing attention due to the need to choose the best model among a diverse set while balancing task reward and exploration cost. Organizations faces decisions like whether to employ a costly API-based LLM or a locally finetuned small LLM, weighing cost against performance. Traditional selection methods often evaluate every candidate model before choosing one, which are becoming impractical given the rising costs of training and finetuning LLMs. Moreover, it is undesirable to allocate excessive resources towards exploring poor-performing models. While some recent works leverage online bandit algorithm to manage such exploration-exploitation trade-off in model selection, they tend to overlook the increasing-then-converging trend in model performances as the model is iteratively finetuned, leading to less accurate predictions and suboptimal model selections. In this paper, we propose a time-increasing bandit algorithm TI-UCB, which effectively predicts the increase of model performances due to finetuning and efficiently balances exploration and exploitation in model selection. To further capture the converging points of models, we develop a change detection mechanism by comparing consecutive increase predictions. We theoretically prove that our algorithm achieves a logarithmic regret upper bound in a typical increasing bandit setting, which implies a fast convergence rate. The advantage of our method is also empirically validated through extensive experiments on classification model selection and online selection of LLMs. Our results highlight the importance of utilizing increasing-then-converging pattern for more efficient and economic model selection in the deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06965v1">Hybrid Human-LLM Corpus Construction and LLM Evaluation for Rare Linguistic Phenomena</a></div>
    <div class="paper-meta">
      📅 2024-03-11
    </div>
    <details class="paper-abstract">
      Argument Structure Constructions (ASCs) are one of the most well-studied construction groups, providing a unique opportunity to demonstrate the usefulness of Construction Grammar (CxG). For example, the caused-motion construction (CMC, ``She sneezed the foam off her cappuccino'') demonstrates that constructions must carry meaning, otherwise the fact that ``sneeze'' in this context causes movement cannot be explained. We form the hypothesis that this remains challenging even for state-of-the-art Large Language Models (LLMs), for which we devise a test based on substituting the verb with a prototypical motion verb. To be able to perform this test at statistically significant scale, in the absence of adequate CxG corpora, we develop a novel pipeline of NLP-assisted collection of linguistically annotated text. We show how dependency parsing and GPT-3.5 can be used to significantly reduce annotation cost and thus enable the annotation of rare phenomena at scale. We then evaluate GPT, Gemini, Llama2 and Mistral models for their understanding of the CMC using the newly collected corpus. We find that all models struggle with understanding the motion component that the CMC adds to a sentence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06660v1">FashionReGen: LLM-Empowered Fashion Report Generation</a></div>
    <div class="paper-meta">
      📅 2024-03-11
    </div>
    <details class="paper-abstract">
      Fashion analysis refers to the process of examining and evaluating trends, styles, and elements within the fashion industry to understand and interpret its current state, generating fashion reports. It is traditionally performed by fashion professionals based on their expertise and experience, which requires high labour cost and may also produce biased results for relying heavily on a small group of people. In this paper, to tackle the Fashion Report Generation (FashionReGen) task, we propose an intelligent Fashion Analyzing and Reporting system based the advanced Large Language Models (LLMs), debbed as GPT-FAR. Specifically, it tries to deliver FashionReGen based on effective catwalk analysis, which is equipped with several key procedures, namely, catwalk understanding, collective organization and analysis, and report generation. By posing and exploring such an open-ended, complex and domain-specific task of FashionReGen, it is able to test the general capability of LLMs in fashion domain. It also inspires the explorations of more high-level tasks with industrial significance in other domains. Video illustration and more materials of GPT-FAR can be found in https://github.com/CompFashion/FashionReGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06593v1">Authorship and the Politics and Ethics of LLM Watermarks</a></div>
    <div class="paper-meta">
      📅 2024-03-11
      | 💬 25 pages. Comments welcome
    </div>
    <details class="paper-abstract">
      Recently, watermarking schemes for large language models (LLMs) have been proposed to distinguish text generated by machines and by humans. The present paper explores philosophical, political, and ethical ramifications of implementing and using watermarking schemes. A definition of authorship that includes both machines (LLMs) and humans is proposed to serve as a backdrop. It is argued that private watermarks may provide private companies with sweeping rights to determine authorship, which is incompatible with traditional standards of authorship determination. Then, possible ramifications of the so-called entropy dependence of watermarking mechanisms are explored. It is argued that entropy may vary for different, socially salient groups. This could lead to group dependent rates at which machine generated text is detected. Specifically, groups more interested in low entropy text may face the challenge that it is harder to detect machine generated text that is of interest to them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06591v1">Academically intelligent LLMs are not necessarily socially intelligent</a></div>
    <div class="paper-meta">
      📅 2024-03-11
    </div>
    <details class="paper-abstract">
      The academic intelligence of large language models (LLMs) has made remarkable progress in recent times, but their social intelligence performance remains unclear. Inspired by established human social intelligence frameworks, particularly Daniel Goleman's social intelligence theory, we have developed a standardized social intelligence test based on real-world social scenarios to comprehensively assess the social intelligence of LLMs, termed as the Situational Evaluation of Social Intelligence (SESI). We conducted an extensive evaluation with 13 recent popular and state-of-art LLM agents on SESI. The results indicate the social intelligence of LLMs still has significant room for improvement, with superficially friendliness as a primary reason for errors. Moreover, there exists a relatively low correlation between the social intelligence and academic intelligence exhibited by LLMs, suggesting that social intelligence is distinct from academic intelligence for LLMs. Additionally, while it is observed that LLMs can't ``understand'' what social intelligence is, their social intelligence, similar to that of humans, is influenced by social factors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06586v1">ContextGPT: Infusing LLMs Knowledge into Neuro-Symbolic Activity Recognition Models</a></div>
    <div class="paper-meta">
      📅 2024-03-11
    </div>
    <details class="paper-abstract">
      Context-aware Human Activity Recognition (HAR) is a hot research area in mobile computing, and the most effective solutions in the literature are based on supervised deep learning models. However, the actual deployment of these systems is limited by the scarcity of labeled data that is required for training. Neuro-Symbolic AI (NeSy) provides an interesting research direction to mitigate this issue, by infusing common-sense knowledge about human activities and the contexts in which they can be performed into HAR deep learning classifiers. Existing NeSy methods for context-aware HAR rely on knowledge encoded in logic-based models (e.g., ontologies) whose design, implementation, and maintenance to capture new activities and contexts require significant human engineering efforts, technical knowledge, and domain expertise. Recent works show that pre-trained Large Language Models (LLMs) effectively encode common-sense knowledge about human activities. In this work, we propose ContextGPT: a novel prompt engineering approach to retrieve from LLMs common-sense knowledge about the relationship between human activities and the context in which they are performed. Unlike ontologies, ContextGPT requires limited human effort and expertise. An extensive evaluation carried out on two public datasets shows how a NeSy model obtained by infusing common-sense knowledge from ContextGPT is effective in data scarcity scenarios, leading to similar (and sometimes better) recognition rates than logic-based approaches with a fraction of the effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14151v2">True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-03-11
      | 💬 Accepted by ICLR2024
    </div>
    <details class="paper-abstract">
      Despite the impressive performance across numerous tasks, large language models (LLMs) often fail in solving simple decision-making tasks due to the misalignment of the knowledge in LLMs with environments. On the contrary, reinforcement learning (RL) agents learn policies from scratch, which makes them always align with environments but difficult to incorporate prior knowledge for efficient explorations. To narrow the gap, we propose TWOSOME, a novel general online framework that deploys LLMs as decision-making agents to efficiently interact and align with embodied environments via RL without requiring any prepared datasets or prior knowledge of the environments. Firstly, we query the joint probabilities of each valid action with LLMs to form behavior policies. Then, to enhance the stability and robustness of the policies, we propose two normalization methods and summarize four prompt design principles. Finally, we design a novel parameter-efficient training architecture where the actor and critic share one frozen LLM equipped with low-rank adapters (LoRA) updated by PPO. We conduct extensive experiments to evaluate TWOSOME. i) TWOSOME exhibits significantly better sample efficiency and performance compared to the conventional RL method, PPO, and prompt tuning method, SayCan, in both classical decision-making environment, Overcooked, and simulated household environment, VirtualHome. ii) Benefiting from LLMs' open-vocabulary feature, TWOSOME shows superior generalization ability to unseen tasks. iii) Under our framework, there is no significant loss of the LLMs' original ability during online PPO finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06354v1">Amharic LLaMA and LLaVA: Multimodal LLMs for Low Resource Languages</a></div>
    <div class="paper-meta">
      📅 2024-03-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) like GPT-4 and LLaMA have shown incredible proficiency at natural language processing tasks and have even begun to excel at tasks across other modalities such as vision and audio. Despite their success, LLMs often struggle to perform well on low-resource languages because there is so little training data available. This shortcoming is especially prevalent with open source models. In this work, we explore training LLaMA-2 to speak Amharic, a language which is spoken by over 50 million people world wide, but has orders of magnitude less data available than languages like English. We employ methods previously used for training LLMs on other languages with data scarcity, and use open source translation models to perform data augmentation and grow our dataset from millions of tokens to billions. We further enhance the capabilities of our model by connecting an image encoder and training on a translated visual instruction tuning dataset in the same manner as LLaVA, resulting in a multimodal Amharic LLM that can understand images along with text. We introduce an Amharic version of a popular benchmarking dataset to evaluate our work. Our models and dataset are open sourced and available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.04963v3">LLM4VV: Developing LLM-Driven Testsuite for Compiler Validation</a></div>
    <div class="paper-meta">
      📅 2024-03-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are a new and powerful tool for a wide span of applications involving natural language and demonstrate impressive code generation abilities. The goal of this work is to automatically generate tests and use these tests to validate and verify compiler implementations of a directive-based parallel programming paradigm, OpenACC. To do so, in this paper, we explore the capabilities of state-of-the-art LLMs, including open-source LLMs -- Meta Codellama, Phind fine-tuned version of Codellama, Deepseek Deepseek Coder and closed-source LLMs -- OpenAI GPT-3.5-Turbo and GPT-4-Turbo. We further fine-tuned the open-source LLMs and GPT-3.5-Turbo using our own testsuite dataset along with using the OpenACC specification. We also explored these LLMs using various prompt engineering techniques that include code template, template with retrieval-augmented generation (RAG), one-shot example, one-shot with RAG, expressive prompt with code template and RAG. This paper highlights our findings from over 5000 tests generated via all the above mentioned methods. Our contributions include: (a) exploring the capabilities of the latest and relevant LLMs for code generation, (b) investigating fine-tuning and prompt methods, and (c) analyzing the outcome of LLMs generated tests including manually analysis of representative set of tests. We found the LLM Deepseek-Coder-33b-Instruct produced the most passing tests followed by GPT-4-Turbo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11998v4">LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset</a></div>
    <div class="paper-meta">
      📅 2024-03-10
    </div>
    <details class="paper-abstract">
      Studying how people interact with large language models (LLMs) in real-world scenarios is increasingly important due to their widespread use in various applications. In this paper, we introduce LMSYS-Chat-1M, a large-scale dataset containing one million real-world conversations with 25 state-of-the-art LLMs. This dataset is collected from 210K unique IP addresses in the wild on our Vicuna demo and Chatbot Arena website. We offer an overview of the dataset's content, including its curation process, basic statistics, and topic distribution, highlighting its diversity, originality, and scale. We demonstrate its versatility through four use cases: developing content moderation models that perform similarly to GPT-4, building a safety benchmark, training instruction-following models that perform similarly to Vicuna, and creating challenging benchmark questions. We believe that this dataset will serve as a valuable resource for understanding and advancing LLM capabilities. The dataset is publicly available at https://huggingface.co/datasets/lmsys/lmsys-chat-1m.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06254v1">LLMs Still Can't Avoid Instanceof: An Investigation Into GPT-3.5, GPT-4 and Bard's Capacity to Handle Object-Oriented Programming Assignments</a></div>
    <div class="paper-meta">
      📅 2024-03-10
      | 💬 This is the author's version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record was published in the Proceedings of the 46th International Conference on Software Engineering: Software Engineering Education and Training track (ICSE-SEET '24, Lisbon, Portugal)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as promising tools to assist students while solving programming assignments. However, object-oriented programming (OOP), with its inherent complexity involving the identification of entities, relationships, and responsibilities, is not yet mastered by these tools. Contrary to introductory programming exercises, there exists a research gap with regard to the behavior of LLMs in OOP contexts. In this study, we experimented with three prominent LLMs - GPT-3.5, GPT-4, and Bard - to solve real-world OOP exercises used in educational settings, subsequently validating their solutions using an Automatic Assessment Tool (AAT). The findings revealed that while the models frequently achieved mostly working solutions to the exercises, they often overlooked the best practices of OOP. GPT-4 stood out as the most proficient, followed by GPT-3.5, with Bard trailing last. We advocate for a renewed emphasis on code quality when employing these models and explore the potential of pairing LLMs with AATs in pedagogical settings. In conclusion, while GPT-4 showcases promise, the deployment of these models in OOP education still mandates supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06221v1">TRAD: Enhancing LLM Agents with Step-Wise Thought Retrieval and Aligned Decision</a></div>
    <div class="paper-meta">
      📅 2024-03-10
      | 💬 Codes available at: https://github.com/skyriver-2000/TRAD-Official
    </div>
    <details class="paper-abstract">
      Numerous large language model (LLM) agents have been built for different tasks like web navigation and online shopping due to LLM's wide knowledge and text-understanding ability. Among these works, many of them utilize in-context examples to achieve generalization without the need for fine-tuning, while few of them have considered the problem of how to select and effectively utilize these examples. Recently, methods based on trajectory-level retrieval with task meta-data and using trajectories as in-context examples have been proposed to improve the agent's overall performance in some sequential decision making tasks. However, these methods can be problematic due to plausible examples retrieved without task-specific state transition dynamics and long input with plenty of irrelevant context. In this paper, we propose a novel framework (TRAD) to address these issues. TRAD first conducts Thought Retrieval, achieving step-level demonstration selection via thought matching, leading to more helpful demonstrations and less irrelevant input noise. Then, TRAD introduces Aligned Decision, complementing retrieved demonstration steps with their previous or subsequent steps, which enables tolerance for imperfect thought and provides a choice for balance between more context and less noise. Extensive experiments on ALFWorld and Mind2Web benchmarks show that TRAD not only outperforms state-of-the-art models but also effectively helps in reducing noise and promoting generalization. Furthermore, TRAD has been deployed in real-world scenarios of a global business insurance company and improves the success rate of robotic process automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06201v1">Are You Being Tracked? Discover the Power of Zero-Shot Trajectory Tracing with LLMs!</a></div>
    <div class="paper-meta">
      📅 2024-03-10
    </div>
    <details class="paper-abstract">
      There is a burgeoning discussion around the capabilities of Large Language Models (LLMs) in acting as fundamental components that can be seamlessly incorporated into Artificial Intelligence of Things (AIoT) to interpret complex trajectories. This study introduces LLMTrack, a model that illustrates how LLMs can be leveraged for Zero-Shot Trajectory Recognition by employing a novel single-prompt technique that combines role-play and think step-by-step methodologies with unprocessed Inertial Measurement Unit (IMU) data. We evaluate the model using real-world datasets designed to challenge it with distinct trajectories characterized by indoor and outdoor scenarios. In both test scenarios, LLMTrack not only meets but exceeds the performance benchmarks set by traditional machine learning approaches and even contemporary state-of-the-art deep learning models, all without the requirement of training on specialized datasets. The results of our research suggest that, with strategically designed prompts, LLMs can tap into their extensive knowledge base and are well-equipped to analyze raw sensor data with remarkable effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06158v1">Are LLMs ready for Visualization?</a></div>
    <div class="paper-meta">
      📅 2024-03-10
      | 💬 10 pages, 8 figures, To appear in IEEE PacificVis Workshop Vis Meets AI
    </div>
    <details class="paper-abstract">
      Generative models have received a lot of attention in many areas of academia and the industry. Their capabilities span many areas, from the invention of images given a prompt to the generation of concrete code to solve a certain programming issue. These two paradigmatic cases fall within two distinct categories of requirements, ranging from "creativity" to "precision", as characterized by Bing Chat, which employs ChatGPT-4 as its backbone. Visualization practitioners and researchers have wondered to what end one of such systems could accomplish our work in a more efficient way. Several works in the literature have utilized them for the creation of visualizations. And some tools such as Lida, incorporate them as part of their pipeline. Nevertheless, to the authors' knowledge, no systematic approach for testing their capabilities has been published, which includes both extensive and in-depth evaluation. Our goal is to fill that gap with a systematic approach that analyzes three elements: whether Large Language Models are capable of correctly generating a large variety of charts, what libraries they can deal with effectively, and how far we can go to configure individual charts. To achieve this objective, we initially selected a diverse set of charts, which are commonly utilized in data visualization. We then developed a set of generic prompts that could be used to generate them, and analyzed the performance of different LLMs and libraries. The results include both the set of prompts and the data sources, as well as an analysis of the performance with different configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06144v1">Simulating Family Conversations using LLMs: Demonstration of Parenting Styles</a></div>
    <div class="paper-meta">
      📅 2024-03-10
    </div>
    <details class="paper-abstract">
      This study presents a framework for conducting psychological and linguistic research through simulated conversations using large language models (LLMs). The proposed methodology offers significant advantages, particularly for simulating human interactions involving potential unethical language or behaviors that would be impermissible in traditional experiments with human participants. As a demonstration, we employed LLMs to simulate family conversations across four parenting styles (authoritarian, authoritative, permissive, and uninvolved). In general, we observed that the characteristics of the four parenting styles were portrayed in the simulated conversations. Several strategies could be used to improve the simulation quality, such as including context awareness, employing a few-shot prompting approach or fine-tuning models to cater to specific simulation requirements. Overall, this study introduces a promising methodology for conducting psychological and linguistic research through simulated conversations, while acknowledging the current limitations and proposing potential solutions for future refinement and improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06070v1">Reframe Anything: LLM Agent for Open World Video Reframing</a></div>
    <div class="paper-meta">
      📅 2024-03-10
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The proliferation of mobile devices and social media has revolutionized content dissemination, with short-form video becoming increasingly prevalent. This shift has introduced the challenge of video reframing to fit various screen aspect ratios, a process that highlights the most compelling parts of a video. Traditionally, video reframing is a manual, time-consuming task requiring professional expertise, which incurs high production costs. A potential solution is to adopt some machine learning models, such as video salient object detection, to automate the process. However, these methods often lack generalizability due to their reliance on specific training data. The advent of powerful large language models (LLMs) open new avenues for AI capabilities. Building on this, we introduce Reframe Any Video Agent (RAVA), a LLM-based agent that leverages visual foundation models and human instructions to restructure visual content for video reframing. RAVA operates in three stages: perception, where it interprets user instructions and video content; planning, where it determines aspect ratios and reframing strategies; and execution, where it invokes the editing tools to produce the final video. Our experiments validate the effectiveness of RAVA in video salient object detection and real-world reframing tasks, demonstrating its potential as a tool for AI-powered video editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.16119v2">ModuLoRA: Finetuning 2-Bit LLMs on Consumer GPUs by Integrating with Modular Quantizers</a></div>
    <div class="paper-meta">
      📅 2024-03-10
      | 💬 Update since being accepted to TMLR. Updated 2Bit results
    </div>
    <details class="paper-abstract">
      We propose a memory-efficient finetuning algorithm for large language models (LLMs) that supports finetuning LLMs with 65B parameters in 2/3/4-bit precision on as little as one 24GB GPU. Our method, modular low-rank adaptation (ModuLoRA), integrates any user-specified weight quantizer with finetuning via low-rank adapters (LoRAs). Our approach relies on a simple quantization-agnostic backward pass that adaptively materializes low-precision LLM weights from a custom black-box quantization module. This approach enables finetuning 2-bit and 3-bit LLMs for the first time -- leveraging state-of-the-art 2-bit QuIP\# quantization and 3-bit OPTQ quantization -- outperforming finetuning that relies on less sophisticated 4-bit and 8-bit methods. In our experiments, \lplora~attains competitive performance on text classification, natural language inference, and instruction following tasks using significantly less memory than existing approaches, and we also surpass the state-of-the-art ROUGE score on a popular summarization task. We release \lplora~together with a series of low-precision models as part of \llmtune, a user-friendly library for quantizing, running, and finetuning LLMs on consumer GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04852v2">Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning</a></div>
    <div class="paper-meta">
      📅 2024-03-10
    </div>
    <details class="paper-abstract">
      In this study, we present aLLM4TS, an innovative framework that adapts Large Language Models (LLMs) for time-series representation learning. Central to our approach is that we reconceive time-series forecasting as a self-supervised, multi-patch prediction task, which, compared to traditional contrastive learning or mask-and-reconstruction methods, captures temporal dynamics in patch representations more effectively. Our strategy encompasses two-stage training: (i). a causal continual pre-training phase on various time-series datasets, anchored on next patch prediction, effectively syncing LLM capabilities with the intricacies of time-series data; (ii). fine-tuning for multi-patch prediction in the targeted time-series context. A distinctive element of our framework is the patch-wise decoding layer, which departs from previous methods reliant on sequence-level decoding. Such a design directly transposes individual patches into temporal sequences, thereby significantly bolstering the model's proficiency in mastering temporal patch-based representations. aLLM4TS demonstrates superior performance in several downstream tasks, proving its effectiveness in deriving temporal representations with enhanced transferability and marking a pivotal advancement in the adaptation of LLMs for time-series analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.15272v4">AutoDroid: LLM-powered Task Automation in Android</a></div>
    <div class="paper-meta">
      📅 2024-03-09
      | 💬 Published in MobiCom 2024; Original title: "Empowering LLM to use Smartphone for Intelligent Task Automation"
    </div>
    <details class="paper-abstract">
      Mobile task automation is an attractive technique that aims to enable voice-based hands-free user interaction with smartphones. However, existing approaches suffer from poor scalability due to the limited language understanding ability and the non-trivial manual efforts required from developers or end-users. The recent advance of large language models (LLMs) in language understanding and reasoning inspires us to rethink the problem from a model-centric perspective, where task preparation, comprehension, and execution are handled by a unified language model. In this work, we introduce AutoDroid, a mobile task automation system capable of handling arbitrary tasks on any Android application without manual efforts. The key insight is to combine the commonsense knowledge of LLMs and domain-specific knowledge of apps through automated dynamic analysis. The main components include a functionality-aware UI representation method that bridges the UI with the LLM, exploration-based memory injection techniques that augment the app-specific domain knowledge of LLM, and a multi-granularity query optimization module that reduces the cost of model inference. We integrate AutoDroid with off-the-shelf LLMs including online GPT-4/GPT-3.5 and on-device Vicuna, and evaluate its performance on a new benchmark for memory-augmented Android task automation with 158 common tasks. The results demonstrated that AutoDroid is able to precisely generate actions with an accuracy of 90.9%, and complete tasks with a success rate of 71.3%, outperforming the GPT-4-powered baselines by 36.4% and 39.7%. The demo, benchmark suites, and source code of AutoDroid will be released at url{https://autodroid-sys.github.io/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05821v1">Optimizing LLM Queries in Relational Workloads</a></div>
    <div class="paper-meta">
      📅 2024-03-09
    </div>
    <details class="paper-abstract">
      Analytical database providers (e.g., Redshift, Databricks, BigQuery) have rapidly added support for invoking Large Language Models (LLMs) through native user-defined functions (UDFs) to help users perform natural language tasks, such as classification, entity extraction, and translation, inside analytical workloads. For instance, an analyst might want to extract customer sentiments on millions of product reviews. However, LLM inference is highly expensive in both computational and economic terms: for example, an NVIDIA L4 GPU running Llama2-7B can only process 6 KB of text per second. In this paper, we explore how to optimize LLM inference for analytical workloads that invoke LLMs within relational queries. We show that relational queries present novel opportunities for accelerating LLM inference, including reordering rows to maximize key-value (KV) cache reuse within the LLM inference engine, reordering columns within a row to further increase cache reuse, and deduplicating redundant inference requests. We implement these optimizations in Apache Spark, with vLLM as the model serving backend and achieve up to 4.4x improvement in end-to-end latency on a benchmark of diverse LLM-based queries on real datasets. To the best of our knowledge, this is the first work to explicitly address the problem of optimizing LLM invocations within SQL queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05636v1">Tuning-Free Accountable Intervention for LLM Deployment -- A Metacognitive Approach</a></div>
    <div class="paper-meta">
      📅 2024-03-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have catalyzed transformative advances across a spectrum of natural language processing tasks through few-shot or zero-shot prompting, bypassing the need for parameter tuning. While convenient, this modus operandi aggravates ``hallucination'' concerns, particularly given the enigmatic ``black-box'' nature behind their gigantic model sizes. Such concerns are exacerbated in high-stakes applications (e.g., healthcare), where unaccountable decision errors can lead to devastating consequences. In contrast, human decision-making relies on nuanced cognitive processes, such as the ability to sense and adaptively correct misjudgments through conceptual understanding. Drawing inspiration from human cognition, we propose an innovative \textit{metacognitive} approach, dubbed \textbf{CLEAR}, to equip LLMs with capabilities for self-aware error identification and correction. Our framework facilitates the construction of concept-specific sparse subnetworks that illuminate transparent decision pathways. This provides a novel interface for model \textit{intervention} after deployment. Our intervention offers compelling advantages: (\textit{i})~at deployment or inference time, our metacognitive LLMs can self-consciously identify potential mispredictions with minimum human involvement, (\textit{ii})~the model has the capability to self-correct its errors efficiently, obviating the need for additional tuning, and (\textit{iii})~the rectification procedure is not only self-explanatory but also user-friendly, enhancing the interpretability and accessibility of the model. By integrating these metacognitive features, our approach pioneers a new path toward engendering greater trustworthiness and accountability in the deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.04235v3">Can LLMs Follow Simple Rules?</a></div>
    <div class="paper-meta">
      📅 2024-03-08
      | 💬 Project website: https://eecs.berkeley.edu/~normanmu/llm_rules; revised content
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05307v1">Tapilot-Crossing: Benchmarking and Evolving LLMs Towards Interactive Data Analysis Agents</a></div>
    <div class="paper-meta">
      📅 2024-03-08
      | 💬 30 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Interactive Data Analysis, the collaboration between humans and LLM agents, enables real-time data exploration for informed decision-making. The challenges and costs of collecting realistic interactive logs for data analysis hinder the quantitative evaluation of Large Language Model (LLM) agents in this task. To mitigate this issue, we introduce Tapilot-Crossing, a new benchmark to evaluate LLM agents on interactive data analysis. Tapilot-Crossing contains 1024 interactions, covering 4 practical scenarios: Normal, Action, Private, and Private Action. Notably, Tapilot-Crossing is constructed by an economical multi-agent environment, Decision Company, with few human efforts. We evaluate popular and advanced LLM agents in Tapilot-Crossing, which underscores the challenges of interactive data analysis. Furthermore, we propose Adaptive Interaction Reflection (AIR), a self-generated reflection strategy that guides LLM agents to learn from successful history. Experiments demonstrate that Air can evolve LLMs into effective interactive data analysis agents, achieving a relative performance improvement of up to 44.5%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05135v1">ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment</a></div>
    <div class="paper-meta">
      📅 2024-03-08
      | 💬 Project Page: https://ella-diffusion.github.io/
    </div>
    <details class="paper-abstract">
      Diffusion models have demonstrated remarkable performance in the domain of text-to-image generation. However, most widely used models still employ CLIP as their text encoder, which constrains their ability to comprehend dense prompts, encompassing multiple objects, detailed attributes, complex relationships, long-text alignment, etc. In this paper, we introduce an Efficient Large Language Model Adapter, termed ELLA, which equips text-to-image diffusion models with powerful Large Language Models (LLM) to enhance text alignment without training of either U-Net or LLM. To seamlessly bridge two pre-trained models, we investigate a range of semantic alignment connector designs and propose a novel module, the Timestep-Aware Semantic Connector (TSC), which dynamically extracts timestep-dependent conditions from LLM. Our approach adapts semantic features at different stages of the denoising process, assisting diffusion models in interpreting lengthy and intricate prompts over sampling timesteps. Additionally, ELLA can be readily incorporated with community models and tools to improve their prompt-following capabilities. To assess text-to-image models in dense prompt following, we introduce Dense Prompt Graph Benchmark (DPG-Bench), a challenging benchmark consisting of 1K dense prompts. Extensive experiments demonstrate the superiority of ELLA in dense prompt following compared to state-of-the-art methods, particularly in multiple object compositions involving diverse attributes and relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10838v2">Rambler: Supporting Writing With Speech via LLM-Assisted Gist Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-03-08
      | 💬 To appear at ACM CHI 2024
    </div>
    <details class="paper-abstract">
      Dictation enables efficient text input on mobile devices. However, writing with speech can produce disfluent, wordy, and incoherent text and thus requires heavy post-processing. This paper presents Rambler, an LLM-powered graphical user interface that supports gist-level manipulation of dictated text with two main sets of functions: gist extraction and macro revision. Gist extraction generates keywords and summaries as anchors to support the review and interaction with spoken text. LLM-assisted macro revisions allow users to respeak, split, merge and transform dictated text without specifying precise editing locations. Together they pave the way for interactive dictation and revision that help close gaps between spontaneous spoken words and well-structured writing. In a comparative study with 12 participants performing verbal composition tasks, Rambler outperformed the baseline of a speech-to-text editor + ChatGPT, as it better facilitates iterative revisions with enhanced user control over the content while supporting surprisingly diverse user strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04960v1">SecGPT: An Execution Isolation Architecture for LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2024-03-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of the natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we propose SecGPT, an architecture for LLM-based systems that aims to mitigate the security and privacy issues that arise with the execution of third-party apps. SecGPT's key idea is to isolate the execution of apps and more precisely mediate their interactions outside of their isolated environments. We evaluate SecGPT against a number of case study attacks and demonstrate that it protects against many security, privacy, and safety issues that exist in non-isolated LLM-based systems. The performance overhead incurred by SecGPT to improve security is under 0.3x for three-quarters of the tested queries. To foster follow-up research, we release SecGPT's source code at https://github.com/llm-platform-security/SecGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04746v1">LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error</a></div>
    <div class="paper-meta">
      📅 2024-03-07
      | 💬 Code and data available at https://github.com/microsoft/simulated-trial-and-error
    </div>
    <details class="paper-abstract">
      Tools are essential for large language models (LLMs) to acquire up-to-date information and take consequential actions in external environments. Existing work on tool-augmented LLMs primarily focuses on the broad coverage of tools and the flexibility of adding new tools. However, a critical aspect that has surprisingly been understudied is simply how accurately an LLM uses tools for which it has been trained. We find that existing LLMs, including GPT-4 and open-source LLMs specifically fine-tuned for tool use, only reach a correctness rate in the range of 30% to 60%, far from reliable use in practice. We propose a biologically inspired method for tool-augmented LLMs, simulated trial and error (STE), that orchestrates three key mechanisms for successful tool use behaviors in the biological system: trial and error, imagination, and memory. Specifically, STE leverages an LLM's 'imagination' to simulate plausible scenarios for using a tool, after which the LLM interacts with the tool to learn from its execution feedback. Both short-term and long-term memory are employed to improve the depth and breadth of the exploration, respectively. Comprehensive experiments on ToolBench show that STE substantially improves tool learning for LLMs under both in-context learning and fine-tuning settings, bringing a boost of 46.7% to Mistral-Instruct-7B and enabling it to outperform GPT-4. We also show effective continual learning of tools via a simple experience replay strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04735v1">SnapNTell: Enhancing Entity-Centric Visual Question Answering with Retrieval Augmented Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-03-07
    </div>
    <details class="paper-abstract">
      Vision-extended LLMs have made significant strides in Visual Question Answering (VQA). Despite these advancements, VLLMs still encounter substantial difficulties in handling queries involving long-tail entities, with a tendency to produce erroneous or hallucinated responses. In this work, we introduce a novel evaluative benchmark named \textbf{SnapNTell}, specifically tailored for entity-centric VQA. This task aims to test the models' capabilities in identifying entities and providing detailed, entity-specific knowledge. We have developed the \textbf{SnapNTell Dataset}, distinct from traditional VQA datasets: (1) It encompasses a wide range of categorized entities, each represented by images and explicitly named in the answers; (2) It features QA pairs that require extensive knowledge for accurate responses. The dataset is organized into 22 major categories, containing 7,568 unique entities in total. For each entity, we curated 10 illustrative images and crafted 10 knowledge-intensive QA pairs. To address this novel task, we devised a scalable, efficient, and transparent retrieval-augmented multimodal LLM. Our approach markedly outperforms existing methods on the SnapNTell dataset, achieving a 66.5\% improvement in the BELURT score. We will soon make the dataset and the source code publicly accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.16344v2">Enabling and Analyzing How to Efficiently Extract Information from Hybrid Long Documents with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate exceptional performance in textual understanding and tabular reasoning tasks. However, their ability to comprehend and analyze hybrid text, containing textual and tabular data, remains underexplored. In this research, we specialize in harnessing the potential of LLMs to comprehend critical information from financial reports, which are hybrid long-documents. We propose an Automated Financial Information Extraction (AFIE) framework that enhances LLMs' ability to comprehend and extract information from financial reports. To evaluate AFIE, we develop a Financial Reports Numerical Extraction (FINE) dataset and conduct an extensive experimental analysis. Our framework is effectively validated on GPT-3.5 and GPT-4, yielding average accuracy increases of 53.94% and 33.77%, respectively, compared to a naive method. These results suggest that the AFIE framework offers accuracy for automated numerical extraction from complex, hybrid documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12226v3">AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling</a></div>
    <div class="paper-meta">
      📅 2024-03-07
      | 💬 28 pages, 16 figures, under review, work in progress
    </div>
    <details class="paper-abstract">
      We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitating any-to-any multimodal conversation while achieving performance comparable to specialized models across all modalities, proving that discrete representations can effectively and conveniently unify multiple modalities within a language model. Demos are shown in https://junzhan2000.github.io/AnyGPT.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.19523v5">Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning</a></div>
    <div class="paper-meta">
      📅 2024-03-07
      | 💬 In Proceedings of ICLR 2024
    </div>
    <details class="paper-abstract">
      Representation learning on text-attributed graphs (TAGs) has become a critical research problem in recent years. A typical example of a TAG is a paper citation graph, where the text of each paper serves as node attributes. Initial graph neural network (GNN) pipelines handled these text attributes by transforming them into shallow or hand-crafted features, such as skip-gram or bag-of-words features. Recent efforts have focused on enhancing these pipelines with language models (LMs), which typically demand intricate designs and substantial computational resources. With the advent of powerful large language models (LLMs) such as GPT or Llama2, which demonstrate an ability to reason and to utilize general knowledge, there is a growing need for techniques which combine the textual modelling abilities of LLMs with the structural learning capabilities of GNNs. Hence, in this work, we focus on leveraging LLMs to capture textual information as features, which can be used to boost GNN performance on downstream tasks. A key innovation is our use of explanations as features: we prompt an LLM to perform zero-shot classification, request textual explanations for its decision-making process, and design an LLM-to-LM interpreter to translate these explanations into informative features for downstream GNNs. Our experiments demonstrate that our method achieves state-of-the-art results on well-established TAG datasets, including Cora, PubMed, ogbn-arxiv, as well as our newly introduced dataset, tape-arxiv23. Furthermore, our method significantly speeds up training, achieving a 2.88 times improvement over the closest baseline on ogbn-arxiv. Lastly, we believe the versatility of the proposed method extends beyond TAGs and holds the potential to enhance other tasks involving graph-text data. Our codes and datasets are available at: https://github.com/XiaoxinHe/TAPE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04132v1">Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference</a></div>
    <div class="paper-meta">
      📅 2024-03-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have unlocked new capabilities and applications; however, evaluating the alignment with human preferences still poses significant challenges. To address this issue, we introduce Chatbot Arena, an open platform for evaluating LLMs based on human preferences. Our methodology employs a pairwise comparison approach and leverages input from a diverse user base through crowdsourcing. The platform has been operational for several months, amassing over 240K votes. This paper describes the platform, analyzes the data we have collected so far, and explains the tried-and-true statistical methods we are using for efficient and accurate evaluation and ranking of models. We confirm that the crowdsourced questions are sufficiently diverse and discriminating and that the crowdsourced human votes are in good agreement with those of expert raters. These analyses collectively establish a robust foundation for the credibility of Chatbot Arena. Because of its unique value and openness, Chatbot Arena has emerged as one of the most referenced LLM leaderboards, widely cited by leading LLM developers and companies. Our demo is publicly available at \url{https://chat.lmsys.org}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04123v1">Exploring LLM-based Agents for Root Cause Analysis</a></div>
    <div class="paper-meta">
      📅 2024-03-07
    </div>
    <details class="paper-abstract">
      The growing complexity of cloud based software systems has resulted in incident management becoming an integral part of the software development lifecycle. Root cause analysis (RCA), a critical part of the incident management process, is a demanding task for on-call engineers, requiring deep domain knowledge and extensive experience with a team's specific services. Automation of RCA can result in significant savings of time, and ease the burden of incident management on on-call engineers. Recently, researchers have utilized Large Language Models (LLMs) to perform RCA, and have demonstrated promising results. However, these approaches are not able to dynamically collect additional diagnostic information such as incident related logs, metrics or databases, severely restricting their ability to diagnose root causes. In this work, we explore the use of LLM based agents for RCA to address this limitation. We present a thorough empirical evaluation of a ReAct agent equipped with retrieval tools, on an out-of-distribution dataset of production incidents collected at Microsoft. Results show that ReAct performs competitively with strong retrieval and reasoning baselines, but with highly increased factual accuracy. We then extend this evaluation by incorporating discussions associated with incident reports as additional inputs for the models, which surprisingly does not yield significant performance improvements. Lastly, we conduct a case study with a team at Microsoft to equip the ReAct agent with tools that give it access to external diagnostic services that are used by the team for manual RCA. Our results show how agents can overcome the limitations of prior work, and practical considerations for implementing such a system in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03897v1">Fuzzing BusyBox: Leveraging LLM and Crash Reuse for Embedded Bug Unearthing</a></div>
    <div class="paper-meta">
      📅 2024-03-06
    </div>
    <details class="paper-abstract">
      BusyBox, an open-source software bundling over 300 essential Linux commands into a single executable, is ubiquitous in Linux-based embedded devices. Vulnerabilities in BusyBox can have far-reaching consequences, affecting a wide array of devices. This research, driven by the extensive use of BusyBox, delved into its analysis. The study revealed the prevalence of older BusyBox versions in real-world embedded products, prompting us to conduct fuzz testing on BusyBox. Fuzzing, a pivotal software testing method, aims to induce crashes that are subsequently scrutinized to uncover vulnerabilities. Within this study, we introduce two techniques to fortify software testing. The first technique enhances fuzzing by leveraging Large Language Models (LLM) to generate target-specific initial seeds. Our study showed a substantial increase in crashes when using LLM-generated initial seeds, highlighting the potential of LLM to efficiently tackle the typically labor-intensive task of generating target-specific initial seeds. The second technique involves repurposing previously acquired crash data from similar fuzzed targets before initiating fuzzing on a new target. This approach streamlines the time-consuming fuzz testing process by providing crash data directly to the new target before commencing fuzzing. We successfully identified crashes in the latest BusyBox target without conducting traditional fuzzing, emphasizing the effectiveness of LLM and crash reuse techniques in enhancing software testing and improving vulnerability detection in embedded systems. Additionally, manual triaging was performed to identify the nature of crashes in the latest BusyBox.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03645v1">K-Link: Knowledge-Link Graph from LLMs for Enhanced Representation Learning in Multivariate Time-Series Data</a></div>
    <div class="paper-meta">
      📅 2024-03-06
      | 💬 12 pages,7 figures
    </div>
    <details class="paper-abstract">
      Sourced from various sensors and organized chronologically, Multivariate Time-Series (MTS) data involves crucial spatial-temporal dependencies, e.g., correlations among sensors. To capture these dependencies, Graph Neural Networks (GNNs) have emerged as powerful tools, yet their effectiveness is restricted by the quality of graph construction from MTS data. Typically, existing approaches construct graphs solely from MTS signals, which may introduce bias due to a small training dataset and may not accurately represent underlying dependencies. To address this challenge, we propose a novel framework named K-Link, leveraging Large Language Models (LLMs) to encode extensive general knowledge and thereby providing effective solutions to reduce the bias. Leveraging the knowledge embedded in LLMs, such as physical principles, we extract a \textit{Knowledge-Link graph}, capturing vast semantic knowledge of sensors and the linkage of the sensor-level knowledge. To harness the potential of the knowledge-link graph in enhancing the graph derived from MTS data, we propose a graph alignment module, facilitating the transfer of semantic knowledge within the knowledge-link graph into the MTS-derived graph. By doing so, we can improve the graph quality, ensuring effective representation learning with GNNs for MTS data. Extensive experiments demonstrate the efficacy of our approach for superior performance across various MTS-related downstream tasks.
    </details>
</div>
