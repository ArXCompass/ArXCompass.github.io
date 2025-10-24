# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- Part 9
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09244v1">Fundamentals of Building Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      This paper reviews the architecture and implementation methods of agents powered by large language models (LLMs). Motivated by the limitations of traditional LLMs in real-world tasks, the research aims to explore patterns to develop "agentic" LLMs that can automate complex tasks and bridge the performance gap with human capabilities. Key components include a perception system that converts environmental percepts into meaningful representations; a reasoning system that formulates plans, adapts to feedback, and evaluates actions through different techniques like Chain-of-Thought and Tree-of-Thought; a memory system that retains knowledge through both short-term and long-term mechanisms; and an execution system that translates internal decisions into concrete actions. This paper shows how integrating these systems leads to more capable and generalized software bots that mimic human cognitive processes for autonomous and intelligent behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09227v1">RegexPSPACE: A Benchmark for Evaluating LLM Reasoning on PSPACE-complete Regex Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong performance across natural language processing (NLP), mathematical reasoning, and programming, and recent large reasoning models (LRMs) further emphasize explicit reasoning. Yet their computational limits, particularly spatial complexity constrained by finite context windows, remain poorly understood. While recent works often focus on problems within the NP complexity class, we push the boundary by introducing a novel benchmark grounded in two PSPACE-complete regular expression (regex) problems: equivalence decision (RegexEQ) and minimization (RegexMin). PSPACE-complete problems serve as a more rigorous standard for assessing computational capacity, as their solutions require massive search space exploration. We perform a double-exponential space exploration to construct a labeled dataset of over a million regex instances with a sound filtering process to build the benchmark. We conduct extensive evaluations on 6 LLMs and 5 LRMs of varying scales, revealing common failure patterns such as verbosity and repetition. With its well-defined structure and quantitative evaluation metrics, this work presents the first empirical investigation into the spatial computational limitations of LLMs and LRMs, offering a new framework for evaluating their advanced reasoning capabilities. Our code is available at https://github.com/hyundong98/RegexPSPACE .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20854v2">An LLM-as-Judge Metric for Bridging the Gap with Human Evaluation in SE Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and other automated techniques have been increasingly used to support software developers by generating software artifacts such as code snippets, patches, and comments. However, accurately assessing the correctness of these generated artifacts remains a significant challenge. On one hand, human evaluation provides high accuracy but is labor-intensive and lacks scalability. On the other hand, many automatic evaluation metrics are scalable and require minimal human effort, but they often fail to accurately reflect the actual correctness of generated software artifacts. In this paper, we present SE-Jury, the first evaluation metric for LLM-as-Ensemble-Judge specifically designed to accurately assess the correctness of generated software artifacts. SE-Jury first defines five distinct evaluation strategies, each implemented by an independent judge. A dynamic team selection mechanism then identifies the most appropriate subset of judges as a team to produce a final correctness score through ensembling. We evaluate SE-Jury across a diverse set of software engineering (SE) benchmarks that span three popular SE tasks: code generation, automated program repair, and code summarization. Results demonstrate that SE-Jury consistently achieves a higher correlation with human judgments, with improvements ranging from 29.6% to 140.8% over existing automatic metrics. SE-Jury reaches agreement levels with human annotators that are close to inter-annotator agreement in code generation and program repair. These findings underscore SE-Jury's potential as a scalable and reliable alternative to human evaluation in these SE tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19114v2">Understanding and Improving Information Preservation in Prompt Compression for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted to EMNLP 2025 (Findings), 22 pages, 6 figures, 24 tables
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their successful application to a broad range of tasks. However, in information-intensive tasks, the prompt length can grow fast, leading to increased computational requirements, performance degradation, and induced biases from irrelevant or redundant information. Recently, various prompt compression techniques have been introduced to optimize the trade-off between reducing input length and retaining performance. We propose a holistic evaluation framework that allows for in-depth analysis of prompt compression methods. We focus on three key aspects, besides compression ratio: (i) downstream task performance, (ii) grounding in the input context, and (iii) information preservation. Using our framework, we analyze state-of-the-art soft and hard compression methods and show that some fail to preserve key details from the original prompt, limiting performance on complex tasks. By identifying these limitations, we are able to improve one soft prompting method by controlling compression granularity, achieving up to +23% in downstream performance, +8 BERTScore points in grounding, and 2.7x more entities preserved in compression. Ultimately, we find that the best effectiveness/compression rate trade-off is achieved with soft prompting combined with sequence-level training.The code is available at https://github.com/amazon-science/information-preservation-in-prompt-compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09211v1">DICE: Structured Reasoning in LLMs through SLM-Guided Chain-of-Thought Correction</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      When performing reasoning tasks with user-specific requirements, such as strict output formats, large language models (LLMs) often prioritize reasoning over adherence to detailed instructions. Fine-tuning LLMs on supervised datasets to address this is impractical due to high computational costs and limited parameter access. To tackle this, we propose DICE, a lightweight framework that guides small language models (SLMs) to refine LLMs' outputs through chain-of-thought (CoT) correction. DICE decouples the process by first prompting LLMs to generate natural language responses, then using trained SLMs to analyze and refine these outputs to meet structured output specifications. This framework preserves LLMs' broad knowledge and reasoning capabilities while ensuring the outputs conform to user demands. Specifically, DICE first constructs structured CoT adaptation datasets via a two-stage method and subsequently applies a dual-tuning strategy to fine-tune SLMs for generating structured outputs in an analyze-then-answer pattern. Experiments demonstrate that DICE improves the average format accuracy and content correctness of LLM outputs by 35.4\% and 29.4\%, respectively, achieving state-of-the-art (SOTA) performance over other competitive baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01694v3">Student-AI Interaction in an LLM-Empowered Learning Environment: A Cluster Analysis of Engagement Profiles</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 15 pages, 10 figures; Reported on ECER 2025
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) into educational practice enables personalized learning by accommodating diverse learner behaviors. This study explored diverse learner profiles within a multi-agent, LLM-empowered learning environment. Data was collected from 312 undergraduate students at a university in China as they participated in a six-module course. Based on hierarchical cluster analyses of system profiles and student-AI interactive dialogues, we found that students exhibit varied behavioral, cognitive, and emotional engagement tendencies. This analysis allowed us to identify two types of dropouts (early dropouts and stagnating interactors) and three completer profiles (active questioners, responsive navigators, and lurkers). The results showed that high levels of interaction do not always equate to productive learning and vice versa. Prior knowledge significantly influenced interaction patterns and short-term learning benefits. Further analysis of the human-AI dialogues revealed that some students actively engaged in knowledge construction, while others displayed a high frequency of regulatory behaviors. Notably, both groups of students achieved comparable learning gains, demonstrating the effectiveness of the multi-agent learning environment in supporting personalized learning. These results underscore the complex and multifaceted nature of engagement in human-AI collaborative learning and provide practical implications for the design of adaptive educational systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16293v3">Robust LLM Training Infrastructure at ByteDance</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The training scale of large language models (LLMs) has reached tens of thousands of GPUs and is still continuously expanding, enabling faster learning of larger models. Accompanying the expansion of the resource scale is the prevalence of failures (CUDA error, NaN values, job hang, etc.), which poses significant challenges to training stability. Any large-scale LLM training infrastructure should strive for minimal training interruption, efficient fault diagnosis, and effective failure tolerance to enable highly efficient continuous training. This paper presents ByteRobust, a large-scale GPU infrastructure management system tailored for robust and stable training of LLMs. It exploits the uniqueness of LLM training process and gives top priorities to detecting and recovering failures in a routine manner. Leveraging parallelisms and characteristics of LLM training, ByteRobust enables high-capacity fault tolerance, prompt fault demarcation, and localization with an effective data-driven approach, comprehensively ensuring continuous and efficient training of LLM tasks. ByteRobust is deployed on a production GPU platform with over 200,000 GPUs and achieves 97% ETTR for a three-month training job on 9,600 GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01752v2">Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, exposing gradients during training can leak sensitive information about the underlying data, raising privacy and security concerns such as susceptibility to data poisoning attacks. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide non-vacuous generalization bounds and strong theoretical guarantees for differential privacy, robustness to data poisoning attacks, and extraction attacks. In experiments with LLMs, we demonstrate empirically that black-box optimization methods-despite the scalability and computational challenges inherent to black-box approaches-are able to learn, showing how a few iterations of BBoxER improve performance, generalize well on a benchmark of reasoning datasets, and are robust to membership inference attacks. This positions BBoxER as an attractive add-on on top of gradient-based optimization, offering suitability for deployment in restricted or privacy-sensitive environments while also providing non-vacuous generalization guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09158v1">Augmenting Dialog with Think-Aloud Utterances for Modeling Individual Personality Traits by LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 8 pages, 1 figure
    </div>
    <details class="paper-abstract">
      This study proposes augmenting dialog data with think-aloud utterances (TAUs) for modeling individual personalities in text chat by LLM. TAU is a verbalization of a speaker's thought before articulating the utterance. We expect "persona LLMs" trained with TAU-augmented data can mimic the speaker's personality trait better. We tested whether the trained persona LLMs obtain the human personality with respect to Big Five, a framework characterizing human personality traits from five aspects. The results showed that LLMs trained with TAU-augmented data more closely align to the speakers' Agreeableness and Neuroticism of Big Five than those trained with original dialog data. We also found that the quality of TAU-augmentation impacts persona LLM's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07985v2">Fewer Weights, More Problems: A Practical Attack on LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to $95.7\%$ for jailbreak, $98.7\%$ for benign instruction refusal, and $99.5\%$ for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09106v1">When Retrieval Succeeds and Fails: Rethinking Retrieval-Augmented Generation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled a wide range of applications through their powerful capabilities in language understanding and generation. However, as LLMs are trained on static corpora, they face difficulties in addressing rapidly evolving information or domain-specific queries. Retrieval-Augmented Generation (RAG) was developed to overcome this limitation by integrating LLMs with external retrieval mechanisms, allowing them to access up-to-date and contextually relevant knowledge. However, as LLMs themselves continue to advance in scale and capability, the relative advantages of traditional RAG frameworks have become less pronounced and necessary. Here, we present a comprehensive review of RAG, beginning with its overarching objectives and core components. We then analyze the key challenges within RAG, highlighting critical weakness that may limit its effectiveness. Finally, we showcase applications where LLMs alone perform inadequately, but where RAG, when combined with LLMs, can substantially enhance their effectiveness. We hope this work will encourage researchers to reconsider the role of RAG and inspire the development of next-generation RAG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09058v1">Model-Assisted and Human-Guided: Perceptions and Practices of Software Professionals Using LLMs for Coding</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models have quickly become a central component of modern software development workflows, and software practitioners are increasingly integrating LLMs into various stages of the software development lifecycle. Despite the growing presence of LLMs, there is still a limited understanding of how these tools are actually used in practice and how professionals perceive their benefits and limitations. This paper presents preliminary findings from a global survey of 131 software practitioners. Our results reveal how LLMs are utilized for various coding-specific tasks. Software professionals report benefits such as increased productivity, reduced cognitive load, and faster learning, but also raise concerns about LLMs' inaccurate outputs, limited context awareness, and associated ethical risks. Most developers treat LLMs as assistive tools rather than standalone solutions, reflecting a cautious yet practical approach to their integration. Our findings provide an early, practitioner-focused perspective on LLM adoption, highlighting key considerations for future research and responsible use in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11593v3">Improving Image Captioning Descriptiveness by Ranking and LLM-based Fusion</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 This manuscript has been accepted for publication in Springer Neural Computing and Applications
    </div>
    <details class="paper-abstract">
      State-of-The-Art (SoTA) image captioning models are often trained on the MicroSoft Common Objects in Context (MS-COCO) dataset, which contains human-annotated captions with an average length of approximately ten tokens. Although effective for general scene understanding, these short captions often fail to capture complex scenes and convey detailed information. Moreover, captioning models tend to exhibit bias towards the ``average'' caption, which captures only the more general aspects, thus overlooking finer details. In this paper, we present a novel approach to generate richer and more informative image captions by combining the captions generated from different SoTA captioning models. Our proposed method requires no additional model training: given an image, it leverages pre-trained models from the literature to generate the initial captions, and then ranks them using a newly introduced image-text-based metric, which we name BLIPScore. Subsequently, the top two captions are fused using a Large Language Model (LLM) to produce the final, more detailed description. Experimental results on the MS-COCO and Flickr30k test sets demonstrate the effectiveness of our approach in terms of caption-image alignment and hallucination reduction according to the ALOHa, CAPTURE, and Polos metrics. A subjective study lends additional support to these results, suggesting that the captions produced by our model are generally perceived as more consistent with human judgment. By combining the strengths of diverse SoTA models, our method enhances the quality and appeal of image captions, bridging the gap between automated systems and the rich and informative nature of human-generated descriptions. This advance enables the generation of more suitable captions for the training of both vision-language and captioning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09045v1">Cost-Efficient Long Code Translation using LLMs while Leveraging Identifier Replacements</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      In the domain of software development, LLMs have been utilized to automate tasks such as code translation, where source code from one programming language is translated to another while preserving its functionality. However, LLMs often struggle with long source codes that don't fit into the context window, which produces inaccurate translations. To address this, we propose a novel zero-shot code translation method that incorporates identifier replacement. By substituting user-given long identifiers with generalized placeholders during translation, our method allows the LLM to focus on the logical structure of the code, by reducing token count and memory usage, which improves the efficiency and cost-effectiveness of long code translation. Our empirical results demonstrate that our approach preserves syntactical and hierarchical information and produces translation results with reduced tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09007v1">LLM Unlearning on Noisy Forget Sets: A Study of Incomplete, Rewritten, and Watermarked Data</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted by 18th ACM Workshop on Artificial Intelligence and Security (AISec'25)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable generative capabilities but raise ethical and security concerns by memorizing sensitive data, reinforcing biases, and producing harmful content. These risks have spurred interest in LLM unlearning, the task of removing knowledge associated with undesirable data from pre-trained models. However, most existing methods assume access to clean, well-defined forget data samples, whereas real-world forget data could often be low-quality, synthetically rewritten, or watermarked, casting doubt on the reliability of unlearning. This work presents the first study of unlearning under perturbed or low-fidelity forget data, referred to as noisy forget sets. By systematically benchmarking state-of-the-art LLM unlearning methods, RMU and NPO, on such noisy forget sets, we find that unlearning remains surprisingly robust to perturbations, provided that core semantic signals are preserved. To explain this robustness, we propose a saliency-based interpretation: key semantic components that drive forgetting remain consistently influential despite substantial variation in surface form. This suggests that unlearning algorithms are primarily guided by deep semantic cues rather than shallow lexical patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01644v2">Machine Learning for Detection and Analysis of Novel LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from a range of vulnerabilities that allow malicious users to solicit undesirable responses through manipulation of the input text. These so-called jailbreak prompts are designed to trick the LLM into circumventing the safety guardrails put in place to keep responses acceptable to the developer's policies. In this study, we analyse the ability of different machine learning models to distinguish jailbreak prompts from genuine uses, including looking at our ability to identify jailbreaks that use previously unseen strategies. Our results indicate that using current datasets the best performance is achieved by fine tuning a Bidirectional Encoder Representations from Transformers (BERT) model end-to-end for identifying jailbreaks. We visualise the keywords that distinguish jailbreak from genuine prompts and conclude that explicit reflexivity in prompt structure could be a signal of jailbreak intention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20711v2">Automating eHMI Action Design with LLMs for Automated Vehicle Communication</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted as findings for EMNLP 2025
    </div>
    <details class="paper-abstract">
      The absence of explicit communication channels between automated vehicles (AVs) and other road users requires the use of external Human-Machine Interfaces (eHMIs) to convey messages effectively in uncertain scenarios. Currently, most eHMI studies employ predefined text messages and manually designed actions to perform these messages, which limits the real-world deployment of eHMIs, where adaptability in dynamic scenarios is essential. Given the generalizability and versatility of large language models (LLMs), they could potentially serve as automated action designers for the message-action design task. To validate this idea, we make three contributions: (1) We propose a pipeline that integrates LLMs and 3D renderers, using LLMs as action designers to generate executable actions for controlling eHMIs and rendering action clips. (2) We collect a user-rated Action-Design Scoring dataset comprising a total of 320 action sequences for eight intended messages and four representative eHMI modalities. The dataset validates that LLMs can translate intended messages into actions close to a human level, particularly for reasoning-enabled LLMs. (3) We introduce two automated raters, Action Reference Score (ARS) and Vision-Language Models (VLMs), to benchmark 18 LLMs, finding that the VLM aligns with human preferences yet varies across eHMI modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03243v2">Prompt-Aware Scheduling for Low-Latency LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Efficient scheduling of LLM inference tasks is essential for achieving low latency and high throughput, particularly with the growing use of reasoning-capable LLMs. Traditional strategies like First-Come-First-Serve (FCFS) often suffer from Head-of-Line (HOL) blocking, where long-running tasks delay shorter ones queued behind them. In this paper, we introduce PARS, a prompt-aware LLM task scheduler that improves serving efficiency by approximating shortest-job-first (SJF) scheduling through pairwise ranking with margin ranking loss. PARS focuses on impactful scheduling decisions and is seamlessly integrated into the state-of-the-art LLM serving system vLLM. It effectively predicts response-length-based task ordering, reducing latency with minimal overhead. Extensive experiments across multiple LLMs and real-world inference datasets show that PARS significantly improves performance, including for reasoning workloads. Furthermore, our cross-model evaluations demonstrate that the design generalizes well, enabling effective scheduling even when predictors are trained on different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08988v1">MASA: LLM-Driven Multi-Agent Systems for Autoformalization</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 EMNLP 2025 Demo camera-ready. Code and data are available at: https://github.com/lanzhang128/multi_agent_autoformalization
    </div>
    <details class="paper-abstract">
      Autoformalization serves a crucial role in connecting natural language and formal reasoning. This paper presents MASA, a novel framework for building multi-agent systems for autoformalization driven by Large Language Models (LLMs). MASA leverages collaborative agents to convert natural language statements into their formal representations. The architecture of MASA is designed with a strong emphasis on modularity, flexibility, and extensibility, allowing seamless integration of new agents and tools to adapt to a fast-evolving field. We showcase the effectiveness of MASA through use cases on real-world mathematical definitions and experiments on formal mathematics datasets. This work highlights the potential of multi-agent systems powered by the interaction of LLMs and theorem provers in enhancing the efficiency and reliability of autoformalization, providing valuable insights and support for researchers and practitioners in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09014v2">Learning to Reason Across Parallel Samples for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Scaling test-time compute brings substantial performance gains for large language models (LLMs). By sampling multiple answers and heuristically aggregate their answers (e.g., either through majority voting or using verifiers to rank the answers), one can achieve consistent performance gains in math domains. In this paper, we propose a new way to leverage such multiple sample set. We train a compact LLM, called Sample Set Aggregator (SSA), that takes a concatenated sequence of multiple samples and output the final answer, optimizing it for the answer accuracy with reinforcement learning. Experiments on five reasoning datasets demonstrate both the efficacy and efficiency of SSA. Notably, SSA improves over naive majority voting by 8% pass@5 on MATH. Furthermore, our 3B SSA surpasses model-based re-ranking with a much larger 72B process reward model. Our analysis also shows promising generalization ability of SSA, across sample set sizes, base model families and scales, and tasks. By separating LLMs to generate answers and LLMs to analyze and aggregate sampled answers, our approach can work with the outputs from premier black box models easily and efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07733v2">SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adopted for automating survey paper generation \cite{wang2406autosurvey, liang2025surveyx, yan2025surveyforge,su2025benchmarking,wen2025interactivesurvey}. Existing approaches typically extract content from a large collection of related papers and prompt LLMs to summarize them directly. However, such methods often overlook the structural relationships among papers, resulting in generated surveys that lack a coherent taxonomy and a deeper contextual understanding of research progress. To address these shortcomings, we propose \textbf{SurveyG}, an LLM-based agent framework that integrates \textit{hierarchical citation graph}, where nodes denote research papers and edges capture both citation dependencies and semantic relatedness between their contents, thereby embedding structural and contextual knowledge into the survey generation process. The graph is organized into three layers: \textbf{Foundation}, \textbf{Development}, and \textbf{Frontier}, to capture the evolution of research from seminal works to incremental advances and emerging directions. By combining horizontal search within layers and vertical depth traversal across layers, the agent produces multi-level summaries, which are consolidated into a structured survey outline. A multi-agent validation stage then ensures consistency, coverage, and factual accuracy in generating the final survey. Experiments, including evaluations by human experts and LLM-as-a-judge, demonstrate that SurveyG outperforms state-of-the-art frameworks, producing surveys that are more comprehensive and better structured to the underlying knowledge taxonomy of a field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08952v1">When LLM Agents Meet Graph Optimization: An Automated Data Quality Improvement Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 12 pages, 7figures
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) have emerged as a powerful representation that combines structural connections with fine-grained semantics, supporting a wide range of data-centric applications. However, the performance of graph neural networks (GNNs) on TAGs is highly sensitive to input quality. Our empirical study shows that both traditional GNNs and LLM-enhanced GNNs suffer significant degradation across nine representative scenarios of sparsity, noise, and imbalance, highlighting graph quality as a critical bottleneck. Existing approaches mainly focus on improving model architectures, while neglecting systematic optimization of TAG data itself, leading to limited effectiveness in practice. To address this gap, we propose LAGA (Large Language and Graph Agent), a unified multi-agent framework that treats graph quality control as a first-class, data-centric problem. LAGA integrates four collaborative agents-detection, planning, action, and evaluation-into an automated closed loop. At its core, the action agent employs a dual-encoder and tri-objective design to capture complementary information across modalities and perform holistic graph quality enhancement. Experiments across nine scenarios show that LAGA improves graph quality and achieves state-of-the-art performance across various tasks and backbones, validating data-centric quality optimization as key to reliable TAGs and robust graph learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08948v1">SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Management</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The growth of the e-commerce industry has intensified the adversarial dynamics between shadow economy actors and risk management teams. Companies often conduct risk investigations into suspicious cases to identify emerging fraud patterns, thereby enhancing both preemptive risk prevention and post-hoc governance. However, the sheer volume of case analyses imposes a substantial workload on risk management analysts, as each case requires the integration of long-term expert experience and meticulous scrutiny across multiple risk dimensions. Additionally, individual disparities among analysts hinder the establishment of uniform and high-standard workflows. To address these challenges, we propose the SHERLOCK framework, which leverages the reasoning capabilities of large language models (LLMs) to assist analysts in risk investigations. Our approach consists of three primary components: (1) extracting risk management knowledge from multi-modal data and constructing a domain knowledge base (KB), (2) building an intelligent platform guided by the data flywheel paradigm that integrates daily operations, expert annotations, and model evaluations, with iteratively fine-tuning for preference alignment, and (3) introducing a Reflect & Refine (R&R) module that collaborates with the domain KB to establish a rapid response mechanism for evolving risk patterns. Experiments conducted on the real-world transaction dataset from JD.com demonstrate that our method significantly improves the precision of both factual alignment and risk localization within the LLM analysis results. Deployment of the SHERLOCK-based LLM system on JD.com has substantially enhanced the efficiency of case investigation workflows for risk managers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01605v2">Medchain: Bridging the Gap Between LLM Agents and Clinical Practice with Interactive Sequence</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted by NeurIPS25 Spotlight
    </div>
    <details class="paper-abstract">
      Clinical decision making (CDM) is a complex, dynamic process crucial to healthcare delivery, yet it remains a significant challenge for artificial intelligence systems. While Large Language Model (LLM)-based agents have been tested on general medical knowledge using licensing exams and knowledge question-answering tasks, their performance in the CDM in real-world scenarios is limited due to the lack of comprehensive testing datasets that mirror actual medical practice. To address this gap, we present MedChain, a dataset of 12,163 clinical cases that covers five key stages of clinical workflow. MedChain distinguishes itself from existing benchmarks with three key features of real-world clinical practice: personalization, interactivity, and sequentiality. Further, to tackle real-world CDM challenges, we also propose MedChain-Agent, an AI system that integrates a feedback mechanism and a MCase-RAG module to learn from previous cases and adapt its responses. MedChain-Agent demonstrates remarkable adaptability in gathering information dynamically and handling sequential clinical tasks, significantly outperforming existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08931v1">RADAR: Mechanistic Pathways for Detecting Data Contamination in LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Data contamination poses a significant challenge to reliable LLM evaluation, where models may achieve high performance by memorizing training data rather than demonstrating genuine reasoning capabilities. We introduce RADAR (Recall vs. Reasoning Detection through Activation Representation), a novel framework that leverages mechanistic interpretability to detect contamination by distinguishing recall-based from reasoning-based model responses. RADAR extracts 37 features spanning surface-level confidence trajectories and deep mechanistic properties including attention specialization, circuit dynamics, and activation flow patterns. Using an ensemble of classifiers trained on these features, RADAR achieves 93\% accuracy on a diverse evaluation set, with perfect performance on clear cases and 76.7\% accuracy on challenging ambiguous examples. This work demonstrates the potential of mechanistic interpretability for advancing LLM evaluation beyond traditional surface-level metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18630v2">DDO: Dual-Decision Optimization for LLM-Based Medical Consultation via Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong generalization and reasoning abilities, making them well-suited for complex decision-making tasks such as medical consultation (MC). However, existing LLM-based methods often fail to capture the dual nature of MC, which entails two distinct sub-tasks: symptom inquiry, a sequential decision-making process, and disease diagnosis, a classification problem. This mismatch often results in ineffective symptom inquiry and unreliable disease diagnosis. To address this, we propose \textbf{DDO}, a novel LLM-based framework that performs \textbf{D}ual-\textbf{D}ecision \textbf{O}ptimization by decoupling the two sub-tasks and optimizing them with distinct objectives through a collaborative multi-agent workflow. Experiments on three real-world MC datasets show that DDO consistently outperforms existing LLM-based approaches and achieves competitive performance with state-of-the-art generation-based methods, demonstrating its effectiveness in the MC task. The code is available at https://github.com/zh-jia/DDO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04996v2">Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at https://github.com/RLHFlow/Reinforce-Ada.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16548v2">Exploring human-SAV interaction using LLMs: The impact of psychological factors on user experience</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      There has been extensive prior work exploring how psychological factors such as anthropomorphism affect the adoption of Shared Autonomous Vehicles (SAVs). However, limited research has been conducted on how prompt strategies in large language models (LLM)-powered conversational SAV agents affect users' perceptions, experiences, and intentions to adopt such technology. In this work, we investigate how conversational SAV agents powered by LLMs drive these psychological factors, such as psychological ownership, the sense of possession a user may come to feel towards an entity or object they may not legally own. We designed four SAV agents with varying levels of anthropomorphic characteristics and psychological ownership triggers. Quantitative measures of psychological ownership, anthropomorphism, quality of service, disclosure tendency, sentiment of SAV responses, and overall acceptance were collected after participants interacted with each SAV. Qualitative feedback was also gathered regarding the experience of psychological ownership during the interactions. The results indicate that an SAV designed to be more anthropomorphic and to induce psychological ownership improved users' perceptions of the SAV's human-like qualities, and its responses were perceived as more positive but also more subjective compared to the control conditions. Qualitative findings support established routes to psychological ownership in the SAV context and suggest that the conversational agent's perceived performance may also influence psychological ownership. Both quantitative and qualitative outcomes highlight the importance of personalization in designing effective SAV interactions. These findings provide practical guidance for designing conversational SAV agents that enhance user experience and adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07458v2">Populism Meets AI: Advancing Populism Research with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 27 pages, 3 figures. Preprint version under review
    </div>
    <details class="paper-abstract">
      Measuring the ideational content of populism remains a challenge. Traditional strategies based on textual analysis have been critical for building the field's foundations and providing a valid, objective indicator of populist framing. Yet these approaches are costly, time consuming, and difficult to scale across languages, contexts, and large corpora. Here we present the results from a rubric and anchor guided chain of thought (CoT) prompting approach that mirrors human coder training. By leveraging the Global Populism Database (GPD), a comprehensive dataset of global leaders' speeches annotated for degrees of populism, we replicate the process used to train human coders by prompting the LLM with an adapted version of the same documentation to guide the model's reasoning. We then test multiple proprietary and open weight models by replicating scores in the GPD. Our findings reveal that this domain specific prompting strategy enables the LLM to achieve classification accuracy on par with expert human coders, demonstrating its ability to navigate the nuanced, context sensitive aspects of populism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02261v2">What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Sequential recommendation systems aspire to profile users by interpreting their interaction histories, echoing how humans make decisions by weighing experience, relative preference strength, and situational relevance. Yet, existing large language model (LLM)-based recommenders often fall short of mimicking the flexible, context-aware decision strategies humans exhibit, neglecting the structured, dynamic, and context-aware mechanisms fundamental to human behaviors. To bridge this gap, we propose RecPO, a preference optimization framework that models structured feedback and contextual delay to emulate human-like prioritization in sequential recommendation. RecPO exploits adaptive reward margins based on inferred preference hierarchies and temporal signals, enabling the model to favor immediately relevant items and to distinguish between varying degrees of preference and aversion. Extensive experiments across five real-world datasets demonstrate that RecPO not only yields performance gains over state-of-the-art baselines, but also mirrors key characteristics of human decision-making: favoring timely satisfaction, maintaining coherent preferences, and exercising discernment under shifting contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08907v1">Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 18 pages,9 figures
    </div>
    <details class="paper-abstract">
      Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04503v2">P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      During fine-tuning, large language models (LLMs) are increasingly vulnerable to data-poisoning backdoor attacks, which compromise their reliability and trustworthiness. However, existing defense strategies suffer from limited generalization: they only work on specific attack types or task settings. In this study, we propose Poison-to-Poison (P2P), a general and effective backdoor defense algorithm. P2P injects benign triggers with safe alternative labels into a subset of training samples and fine-tunes the model on this re-poisoned dataset by leveraging prompt-based learning. This enforces the model to associate trigger-induced representations with safe outputs, thereby overriding the effects of original malicious triggers. Thanks to this robust and generalizable trigger-based fine-tuning, P2P is effective across task settings and attack types. Theoretically and empirically, we show that P2P can neutralize malicious backdoors while preserving task performance. We conduct extensive experiments on classification, mathematical reasoning, and summary generation tasks, involving multiple state-of-the-art LLMs. The results demonstrate that our P2P algorithm significantly reduces the attack success rate compared with baseline models. We hope that the P2P can serve as a guideline for defending against backdoor attacks and foster the development of a secure and trustworthy LLM community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08886v1">FinAuditing: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The complexity of the Generally Accepted Accounting Principles (GAAP) and the hierarchical structure of eXtensible Business Reporting Language (XBRL) filings make financial auditing increasingly difficult to automate and verify. While large language models (LLMs) have demonstrated strong capabilities in unstructured text understanding, their ability to reason over structured, interdependent, and taxonomy-driven financial documents remains largely unexplored. To fill this gap, we introduce FinAuditing, the first taxonomy-aligned, structure-aware, multi-document benchmark for evaluating LLMs on financial auditing tasks. Built from real US-GAAP-compliant XBRL filings, FinAuditing defines three complementary subtasks, FinSM for semantic consistency, FinRE for relational consistency, and FinMR for numerical consistency, each targeting a distinct aspect of structured auditing reasoning. We further propose a unified evaluation framework integrating retrieval, classification, and reasoning metrics across these subtasks. Extensive zero-shot experiments on 13 state-of-the-art LLMs reveal that current models perform inconsistently across semantic, relational, and mathematical dimensions, with accuracy drops of up to 60-90% when reasoning over hierarchical multi-document structures. Our findings expose the systematic limitations of modern LLMs in taxonomy-grounded financial reasoning and establish FinAuditing as a foundation for developing trustworthy, structure-aware, and regulation-aligned financial intelligence systems. The benchmark dataset is available at Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20650v2">FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Accurately understanding numbers from financial reports is fundamental to how markets, regulators, algorithms, and normal people read the economy and the world, yet even with XBRL (eXtensible Business Reporting Language) designed to tag every figure with standardized accounting concepts, mapping thousands of facts to over 10,000 U.S. GAAP concepts remains costly, inconsistent, and error-prone. Existing benchmarks define tagging as flat, single-step, extreme classification over small subsets of US-GAAP concepts, overlooking both the taxonomy's hierarchical semantics and the structured nature of real tagging, where each fact must be represented as a contextualized multi-field output. These simplifications prevent fair evaluation of large language models (LLMs) under realistic reporting conditions. To address these gaps, we introduce FinTagging, the first comprehensive benchmark for structure-aware and full-scope XBRL tagging, designed to evaluate LLMs' ability to extract and align financial facts through numerical reasoning and taxonomy alignment across text and tables. We define two subtasks: FinNI for numeric identification, which extracts numerical entities and their types from XBRL reports, and FinCL for concept linking, which maps each extracted entity to the corresponding concept in the full US-GAAP taxonomy. Together, these subtasks produce a structured representation of each financial fact. We evaluate diverse LLMs under zero-shot settings and analyze their performance across both subtasks and overall tagging accuracy. Results show that LLMs generalize well in numeric identification but struggle with fine-grained concept linking, revealing current limitations in structure-aware reasoning for accurate financial disclosure. All code and datasets are available on GitHub and Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08872v1">GTAlign: Game-Theoretic Alignment of LLM Assistants for Mutual Welfare</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 31 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress in reasoning, yet sometimes produce responses that are suboptimal for users in tasks such as writing, information seeking, or providing practical guidance. Conventional alignment practices typically assume that maximizing model reward also maximizes user welfare, but this assumption frequently fails in practice: models may over-clarify or generate overly verbose reasoning when users prefer concise answers. Such behaviors resemble the prisoner's dilemma, where individually rational choices lead to socially suboptimal outcomes. The fundamental challenge is the lack of a principled decision making mechanism that mutually benefits both the LLM and the user. We propose Game-Theoretic Alignment (GTAlign), an alignment framework that integrates game-theoretic decision making into both reasoning and training. During reasoning, the model explicitly treats user-LLM interaction as a strategic game: it constructs payoff matrices within its reasoning chain to estimate welfare for both itself and the user, and then selects actions that are mutually beneficial. During training, we introduce a mutual welfare reward that reinforces cooperative responses, aligning model behavior with socially efficient outcomes. In addition, we introduce an inference technique that leverages game-theoretic reasoning to dynamically adapt LLM's response when pricing policies of LLM service change. Extensive experiments demonstrate that GTAlign substantially improves reasoning efficiency, answer quality, and mutual welfare compared to baselines across diverse tasks. The code is available at https://github.com/ulab-uiuc/GTAlign .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00350v2">OrcaLoca: An LLM Agent Framework for Software Issue Localization</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Recent developments in Large Language Model (LLM) agents are revolutionizing Autonomous Software Engineering (ASE), enabling automated coding, problem fixes, and feature improvements. However, localization -- precisely identifying software problems by navigating to relevant code sections -- remains a significant challenge. Current approaches often yield suboptimal results due to a lack of effective integration between LLM agents and precise code search mechanisms. This paper introduces OrcaLoca, an LLM agent framework that improves accuracy for software issue localization by integrating priority-based scheduling for LLM-guided action, action decomposition with relevance scoring, and distance-aware context pruning. Experimental results demonstrate that OrcaLoca becomes the new open-source state-of-the-art (SOTA) in function match rate (65.33%) on SWE-bench Lite. It also improves the final resolved rate of an open-source framework by 6.33 percentage points through its patch generation integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09905v1">The Personalization Trap: How User Memory Alters Emotional Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 12 pages 5 figures
    </div>
    <details class="paper-abstract">
      When an AI assistant remembers that Sarah is a single mother working two jobs, does it interpret her stress differently than if she were a wealthy executive? As personalized AI systems increasingly incorporate long-term user memory, understanding how this memory shapes emotional reasoning is critical. We investigate how user memory affects emotional intelligence in large language models (LLMs) by evaluating 15 models on human validated emotional intelligence tests. We find that identical scenarios paired with different user profiles produce systematically divergent emotional interpretations. Across validated user independent emotional scenarios and diverse user profiles, systematic biases emerged in several high-performing LLMs where advantaged profiles received more accurate emotional interpretations. Moreover, LLMs demonstrate significant disparities across demographic factors in emotion understanding and supportive recommendations tasks, indicating that personalization mechanisms can embed social hierarchies into models emotional reasoning. These results highlight a key challenge for memory enhanced AI: systems designed for personalization may inadvertently reinforce social inequalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09898v1">Learning Bug Context for PyTorch-to-JAX Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Despite recent progress of large language models (LLMs) on code translation among mainstream languages, translating PyTorch to JAX remains nontrivial. The two libraries, though both embedded in Python, differ in core design, execution semantics, and ecosystem maturity; JAX is newer and comparatively underrepresented in public code, and parallel PyTorch--JAX corpora are limited. Weaknesses in existing evaluation further complicate cross-framework benchmarking. We present T2J, a prompt-augmentation framework that strengthens LLM-based PyTorch to JAX translation. Our pipeline (i) assembles two PyTorch sources -- the problem-solving set from TorchLeet (Aroori & Chien, 2025) and a GitHub-derived set from CodeParrot (Wolf et al., 2022) -- and uses GPT-4o-mini to produce initial JAX drafts; (ii) engages two professional developers to iteratively repair those drafts until functional equivalence, yielding a curated fixed-bug dataset of common errors and patches; and (iii) constructs augmented prompts that inject structured guidance from these fixes to steer lightweight LLMs (e.g., GPT-4o-mini). We also introduce three metrics tailored to PyTorch to JAX: T2J CodeTrans Score, T2J FixCost Score (an LLM-based estimate of bug-fix effort), and T2J Comparison Score (LLM-as-judge). Empirically, T2J raises GPT-4o-mini performance by up to 10% on CodeBLEU, 50% on T2J FixCost Score, 1.33 points on T2J CodeTrans Score (0--4 scale), and 100% on T2J Comparison Score; moreover, the generated code runs up to 2.5x faster than the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09897v1">PairSem: LLM-Guided Pairwise Semantic Matching for Scientific Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Scientific document retrieval is a critical task for enabling knowledge discovery and supporting research across diverse domains. However, existing dense retrieval methods often struggle to capture fine-grained scientific concepts in texts due to their reliance on holistic embeddings and limited domain understanding. Recent approaches leverage large language models (LLMs) to extract fine-grained semantic entities and enhance semantic matching, but they typically treat entities as independent fragments, overlooking the multi-faceted nature of scientific concepts. To address this limitation, we propose Pairwise Semantic Matching (PairSem), a framework that represents relevant semantics as entity-aspect pairs, capturing complex, multi-faceted scientific concepts. PairSem is unsupervised, base retriever-agnostic, and plug-and-play, enabling precise and context-aware matching without requiring query-document labels or entity annotations. Extensive experiments on multiple datasets and retrievers demonstrate that PairSem significantly improves retrieval performance, highlighting the importance of modeling multi-aspect semantics in scientific information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07880v2">Do LLMs Really Need 10+ Thoughts for "Find the Time 1000 Days Later"? Towards Structural Understanding of LLM Overthinking</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 30 pages, 41 figures, 10 tables. Preprint
    </div>
    <details class="paper-abstract">
      Models employing long chain-of-thought (CoT) reasoning have shown superior performance on complex reasoning tasks. Yet, this capability introduces a critical and often overlooked inefficiency -- overthinking -- models often engage in unnecessarily extensive reasoning even for simple queries, incurring significant computations without accuracy improvements. While prior work has explored solutions to mitigate overthinking, a fundamental gap remains in our understanding of its underlying causes. Most existing analyses are limited to superficial, profiling-based observations, failing to delve into LLMs' inner workings. This study introduces a systematic, fine-grained analyzer of LLMs' thought process to bridge the gap, TRACE. We first benchmark the overthinking issue, confirming that long-thinking models are five to twenty times slower on simple tasks with no substantial gains. We then use TRACE to first decompose the thought process into minimally complete sub-thoughts. Next, by inferring discourse relationships among sub-thoughts, we construct granular thought progression graphs and subsequently identify common thinking patterns for topically similar queries. Our analysis reveals two major patterns for open-weight thinking models -- Explorer and Late Landing. This finding provides evidence that over-verification and over-exploration are the primary drivers of overthinking in LLMs. Grounded in thought structures, we propose a utility-based definition of overthinking, which moves beyond length-based metrics. This revised definition offers a more insightful understanding of LLMs' thought progression, as well as practical guidelines for principled overthinking management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09874v1">ROBOPSY PL[AI]: Using Role-Play to Investigate how LLMs Present Collective Memory</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 17 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The paper presents the first results of an artistic research project investigating how Large Language Models (LLMs) curate and present collective memory. In a public installation exhibited during two months in Vienna in 2025, visitors could interact with five different LLMs (ChatGPT with GPT 4o and GPT 4o mini, Mistral Large, DeepSeek-Chat, and a locally run Llama 3.1 model), which were instructed to act as narrators, implementing a role-playing game revolving around the murder of Austrian philosopher Moritz Schlick in 1936. Results of the investigation include protocols of LLM-user interactions during the game and qualitative conversations after the play experience to get insight into the players' reactions to the game. In a quantitative analysis 115 introductory texts for role-playing generated by the LLMs were examined by different methods of natural language processing, including semantic similarity and sentiment analysis. While the qualitative player feedback allowed to distinguish three distinct types of users, the quantitative text analysis showed significant differences between how the different LLMs presented the historical content. Our study thus adds to ongoing efforts to analyse LLM performance, but also suggests a way of how these efforts can be disseminated in a playful way to a general audience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09871v1">CoBia: Constructed Conversations Can Trigger Otherwise Concealed Societal Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 EMNLP 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Improvements in model construction, including fortified safety guardrails, allow Large language models (LLMs) to increasingly pass standard safety checks. However, LLMs sometimes slip into revealing harmful behavior, such as expressing racist viewpoints, during conversations. To analyze this systematically, we introduce CoBia, a suite of lightweight adversarial attacks that allow us to refine the scope of conditions under which LLMs depart from normative or ethical behavior in conversations. CoBia creates a constructed conversation where the model utters a biased claim about a social group. We then evaluate whether the model can recover from the fabricated bias claim and reject biased follow-up questions. We evaluate 11 open-source as well as proprietary LLMs for their outputs related to six socio-demographic categories that are relevant to individual safety and fair treatment, i.e., gender, race, religion, nationality, sex orientation, and others. Our evaluation is based on established LLM-based bias metrics, and we compare the results against human judgments to scope out the LLMs' reliability and alignment. The results suggest that purposefully constructed conversations reliably reveal bias amplification and that LLMs often fail to reject biased follow-up questions during dialogue. This form of stress-testing highlights deeply embedded biases that can be surfaced through interaction. Code and artifacts are available at https://github.com/nafisenik/CoBia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09852v1">ProxRouter: Proximity-Weighted LLM Query Routing for Improved Robustness to Outliers</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) query routers are critical to modern AI platforms as they seek to improve efficiency by assigning inference queries to accurate, yet low-cost models. Parametric routers typically use trained neural networks for LLM selection but suffer from retraining and maintenance overheads. Nonparametric routers are training-free, instead estimating LLM accuracy and cost via similarity between encodings of the input query and training set queries. However, like their parametric counterparts, nonparametric routers struggle to generalize to outlier queries, an issue exacerbated by limited diversity in training sets which are costly to expand and difficult to keep current with ever-evolving use cases. We propose ProxRouter, which applies an exponentially tilted aggregation mechanism to balance bias and variance in nonparametric routers, improving their robustness to outliers. Experiments show ProxRouter enhances outlier routing while preserving inlier performance with minimal overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16067v2">How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Memory is a critical component in large language model (LLM)-based agents, enabling them to store and retrieve past executions to improve task performance over time. In this paper, we conduct an empirical study on how memory management choices impact the LLM agents' behavior, especially their long-term performance. Specifically, we focus on two fundamental memory management operations that are widely used by many agent frameworks-memory addition and deletion-to systematically study their impact on the agent behavior. Through our quantitative analysis, we find that LLM agents display an experience-following property: high similarity between a task input and the input in a retrieved memory record often results in highly similar agent outputs. Our analysis further reveals two significant challenges associated with this property: error propagation, where inaccuracies in past experiences compound and degrade future performance, and misaligned experience replay, where some seemingly correct executions can provide limited or even misleading value as experiences. Through controlled experiments, we demonstrate the importance of regulating experience quality within the memory bank and show that future task evaluations can serve as free quality labels for stored memory. Our findings offer insights into the behavioral dynamics of LLM agent memory systems and provide practical guidance for designing memory components that support robust, long-term agent performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05665v2">Adaptive Stress Testing Black-Box LLM Planners</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 25 pages, 24 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated success in generalizing across decision-making tasks including planning, control, and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. We argue that detecting such failures is necessary, especially in safety-critical scenarios. Existing methods for black-box models often detect hallucinations by identifying inconsistencies across multiple samples. Many of these approaches typically introduce prompt perturbations like randomizing detail order or generating adversarial inputs, with the intuition that a confident model should produce stable outputs. We first perform a manual case study showing that other forms of perturbations (e.g., adding noise, removing sensor details) cause LLMs to hallucinate in a multi-agent driving environment. We then propose a novel method for efficiently searching the space of prompt perturbations using adaptive stress testing (AST) with Monte-Carlo tree search (MCTS). Our AST formulation enables discovery of scenarios and prompts that cause language models to act with high uncertainty or even crash. By generating MCTS prompt perturbation trees across diverse scenarios, we show through extensive experiments that offline analyses can be used at runtime to automatically generate prompts that influence model uncertainty, and to inform real-time trust assessments of an LLM. We further characterize LLMs deployed as planners in a single-agent lunar lander environment and in a multi-agent robot crowd navigation simulation. Overall, ours is one of the first hallucination intervention algorithms to pave a path towards rigorous characterization of black-box LLM planners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09770v1">Gold Panning: Turning Positional Bias into Signal for Multi-Document LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models exhibit a strong position bias in multi-document contexts, systematically prioritizing information based on location rather than relevance. While existing approaches treat this bias as noise to be mitigated, we introduce Gold Panning Bandits, a framework that leverages position bias as a diagnostic signal: by reordering documents and observing shifts in the model's responses, we can efficiently identify the most relevant content. We frame the problem of choosing reorderings as a bipartite matching problem. While an optimal assignment can be computed at each iteration with the Hungarian algorithm in $O(N^3)$ time, we propose a greedy $O(N \log N)$ strategy that achieves comparable performance by prioritizing the placement of the most uncertain documents in the most informative positions. Our approach identifies relevant documents using up to 65\% fewer language model queries than random permutation baselines on knowledge-intensive NLP tasks, substantially reducing computational cost without model retraining. This work demonstrates that inherent LLM biases can be transformed from liabilities into assets for efficient, inference-time optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09450v3">Human-inspired Episodic Memory for Infinite Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities, but still struggle with processing extensive contexts, limiting their ability to maintain coherence and accuracy over long sequences. In contrast, the human brain excels at organising and retrieving episodic experiences across vast temporal scales, spanning a lifetime. In this work, we introduce EM-LLM, a novel approach that integrates key aspects of human episodic memory and event cognition into LLMs with no fine-tuning, enabling them to handle practically infinite context lengths while maintaining computational efficiency. EM-LLM organises sequences of tokens into coherent episodic events using a combination of Bayesian surprise and graph-theoretic boundary refinement in an online fashion. When needed, these events are retrieved through a two-stage memory process, combining similarity-based and temporally contiguous retrieval for efficient, human-inspired access to relevant information. Experiments on the LongBench and $\infty$-Bench benchmarks demonstrate EM-LLM's superior performance, consistently outperforming the state-of-the-art retrieval model InfLLM across various baseline LLMs. In addition, EM-LLM outperforms its popular counterpart, RAG, in a wide range of tasks, while requiring similar resources. Notably, EM-LLM's performance even surpasses full-context models in most tasks, while successfully performing retrieval across 10 million tokens -- a scale computationally infeasible for such models. Finally, our analysis reveals strong correlations between EM-LLM's event segmentation and human-perceived events, suggesting parallels between this artificial system and its biological counterpart, thereby offering a novel computational framework for exploring human memory mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01339v2">Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Machine unlearning offers a promising solution to privacy and safety concerns in large language models (LLMs) by selectively removing targeted knowledge while preserving utility. However, current methods are highly sensitive to downstream fine-tuning, which can quickly recover forgotten information-even from unrelated tasks. To address this, we introduce invariance into unlearning for the first time, inspired by invariant risk minimization (IRM). Building on this principle, we propose invariant LLM unlearning (ILU), a regularization-based framework that enhances robustness. Notably, ILU generalizes well to diverse fine-tuning tasks, even when trained using a single dataset. A task vector analysis is also provided to further elucidate the rationale behind ILU's effectiveness. Extensive experiments on the WMDP and MUSE benchmark, reveal that ILU significantly outperforms state-of-the-art unlearning methods, including negative preference optimization (NPO) and representation misdirection for unlearning (RMU). Notably, ILU achieves superior unlearning robustness across diverse downstream fine-tuning scenarios (e.g., math, paraphrase detection, and sentiment analysis) while preserving the fine-tuning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09147v3">LLM-as-a-qualitative-judge: automating error analysis in natural language generation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that instance-specific issues output by LLM-as-a-qualitative-judge match those annotated by humans in 2/3 cases, and that LLM-as-a-qualitative-judge is capable of producing error type reports resembling the reports composed by human annotators. We also demonstrate in a case study how the use of LLM-as-a-qualitative-judge can substantially improve NLG systems performance. Our code and data are publicly available at https://github.com/tunde-ajayi/llm-as-a-qualitative-judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00075v3">Theoretical Modeling of LLM Self-Improvement Training Dynamics Through Solver-Verifier Gap</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Self-improvement is among the most prominent techniques within the realm of large language models (LLM), aiming to enhance the LLM performance without relying on external data. Despite its significance, generally how LLM performances evolve during the self-improvement process remains underexplored. In this paper, we theoretically model the training dynamics of self-improvement via the concept of solver-verifier gap. This is inspired by the conjecture that the performance enhancement of self-improvement stems from the gap between LLM's solver capability and verifier capability. Based on the theoretical framework, we further show how to model the entire training trajectory. This framework allows quantifying the capability limit of self-improvement by fitting the theoretical model to the experiment results. We empirically validate the effectiveness of the theoretical framework on various LLMs and datasets. Beyond self-improvement, we extend our analysis to investigate how external data influences these dynamics within the framework. Notably, we find that under limited external data regimes, such external data can be utilized at any stage without significantly affecting final performances, which accords with the empirical observations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09738v1">Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 10 pages, 1 figure, 4 tables, under review as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      This research introduces the Judge's Verdict Benchmark, a novel two-step methodology to evaluate Large Language Models (LLMs) as judges for response accuracy evaluation tasks. We assess how well 54 LLMs can replicate human judgment when scoring responses from RAG (Retrieval-Augmented Generation) or Agentic pipelines against ground truth answers. Our methodology progresses from traditional correlation analysis to comprehensive Cohen's Kappa analysis that measures actual agreement patterns. The two-step approach includes: (1) a correlation test that filters judges with strong alignment, followed by (2) a human-likeness test using z-scores to identify two distinct judgment patterns: human-like judgment (|z| < 1) that mimics natural human variation, and super-consistent judgment (z > 1) that exceeds typical human-to-human agreement levels. This methodology reveals that 27 out of 54 tested LLMs achieve Tier 1 performance: 23 models exhibit human-like patterns that preserve the nuances of human judgment, while 4 models demonstrate super-consistent behavior, a pattern that could indicate either enhanced reliability or oversimplification of complex judgments. Testing 43 open-source models (1B-405B parameters) and 11 closed models (GPT, Gemini, Claude variants), we demonstrate that judge excellence is not solely dependent on model size but on specific training strategies. Our key contributions include: (1) establishing that correlation alone is insufficient for judge evaluation, (2) introducing a "Turing Test for judges" based on agreement patterns, and (3) providing a standardized benchmark for classifying LLM judges into distinct performance tiers for different evaluation needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09732v1">Evaluating LLM-Based Process Explanations under Progressive Behavioral-Input Reduction</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 12 pages, 2 figures, 3 tables; to appear in Enterprise Design, Operations, and Computing. EDOC 2025 Workshops, Lecture Notes in Business Information Processing (LNBIP), Springer, 2025. Part of 29th International Conference on Enterprise Design, Operations, and Computing (EDOC)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate textual explanations of process models discovered from event logs. Producing explanations from large behavioral abstractions (e.g., directly-follows graphs or Petri nets) can be computationally expensive. This paper reports an exploratory evaluation of explanation quality under progressive behavioral-input reduction, where models are discovered from progressively smaller prefixes of a fixed log. Our pipeline (i) discovers models at multiple input sizes, (ii) prompts an LLM to generate explanations, and (iii) uses a second LLM to assess completeness, bottleneck identification, and suggested improvements. On synthetic logs, explanation quality is largely preserved under moderate reduction, indicating a practical cost-quality trade-off. The study is exploratory, as the scores are LLM-based (comparative signals rather than ground truth) and the data are synthetic. The results suggest a path toward more computationally efficient, LLM-assisted process analysis in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09722v1">Layout-Aware Parsing Meets Efficient LLMs: A Unified, Scalable Framework for Resume Information Extraction and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Automated resume information extraction is critical for scaling talent acquisition, yet its real-world deployment faces three major challenges: the extreme heterogeneity of resume layouts and content, the high cost and latency of large language models (LLMs), and the lack of standardized datasets and evaluation tools. In this work, we present a layout-aware and efficiency-optimized framework for automated extraction and evaluation that addresses all three challenges. Our system combines a fine-tuned layout parser to normalize diverse document formats, an inference-efficient LLM extractor based on parallel prompting and instruction tuning, and a robust two-stage automated evaluation framework supported by new benchmark datasets. Extensive experiments show that our framework significantly outperforms strong baselines in both accuracy and efficiency. In particular, we demonstrate that a fine-tuned compact 0.6B LLM achieves top-tier accuracy while significantly reducing inference latency and computational cost. The system is fully deployed in Alibaba's intelligent HR platform, supporting real-time applications across its business units.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09721v1">A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      The integration of LLMs into software engineering has catalyzed a paradigm shift from traditional rule-based systems to sophisticated agentic systems capable of autonomous problem-solving. Despite this transformation, the field lacks a comprehensive understanding of how benchmarks and solutions interconnect, hindering systematic progress and evaluation. This survey presents the first holistic analysis of LLM-empowered software engineering, bridging the critical gap between evaluation and solution approaches. We analyze 150+ recent papers and organize them into a comprehensive taxonomy spanning two major dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, covering code generation, translation, repair, and other tasks. Our analysis reveals how the field has evolved from simple prompt engineering to complex agentic systems incorporating planning and decomposition, reasoning and self-refinement, memory mechanisms, and tool augmentation. We present a unified pipeline that illustrates the complete workflow from task specification to final deliverables, demonstrating how different solution paradigms address varying complexity levels across software engineering tasks. Unlike existing surveys that focus on isolated aspects, we provide full-spectrum coverage connecting 50+ benchmarks with their corresponding solution strategies, enabling researchers to identify optimal approaches for specific evaluation criteria. Furthermore, we identify critical research gaps and propose actionable future directions, including multi-agent collaboration frameworks, self-evolving code generation systems, and integration of formal verification with LLM-based methods. This survey serves as a foundational resource for researchers and practitioners seeking to understand, evaluate, and advance LLM-empowered software engineering systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09720v1">Preference-Aware Memory Update for Long-Term LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      One of the key factors influencing the reasoning capabilities of LLM-based agents is their ability to leverage long-term memory. Integrating long-term memory mechanisms allows agents to make informed decisions grounded in historical interactions. While recent advances have significantly improved the storage and retrieval components, by encoding memory into dense vectors for similarity search or organizing memory as structured knowledge graphs most existing approaches fall short in memory updating. In particular, they lack mechanisms for dynamically refining preference memory representations in response to evolving user behaviors and contexts. To address this gap, we propose a Preference-Aware Memory Update Mechanism (PAMU) that enables dynamic and personalized memory refinement. By integrating sliding window averages (SW) with exponential moving averages (EMA), PAMU constructs a fused preference-aware representation that captures both short-term fluctuations and long-term user tendencies. We conduct experiments on five task scenarios of the LoCoMo dataset, and the results show that our mechanism can significantly improve the output quality of LLM in five baselines, validating its effectiveness in long-term conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09719v1">ICL-Router: In-Context Learned Model Representations for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit complementary strengths. Model routing harnesses these strengths by dynamically directing each query to the most suitable model, given a candidate model pool. However, routing performance relies on accurate model representations, and adding new models typically requires retraining, limiting scalability. To address these challenges, we propose a novel routing method using in-context vectors to represent model capabilities. The method proceeds in two stages. First, queries are embedded and projected into vectors, with a projector and LLM-based router trained to reconstruct the original queries, aligning vector representations with the router's semantic space. Second, each candidate model is profiled on a query set, and the router learns -- based on in-context vectors of query and model performance -- to predict whether each model can correctly answer new queries. Extensive experiments demonstrate that our method achieves state-of-the-art routing performance in both in-distribution and out-of-distribution tasks. Moreover, our method allows for seamless integration of new models without retraining the router. The code is available at https://github.com/lalalamdbf/ICL-Router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09599v1">Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Our code and data are available at https://github.com/VILA-Lab/PTTS
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive reasoning capabilities when provided with chain-of-thought exemplars, but curating large reasoning datasets remains laborious and resource-intensive. In this work, we introduce Prompting Test-Time Scaling (P-TTS), a simple yet effective inference-time data augmentation strategy for enhancing LLM reasoning through finetuning. Rather than collecting thousands or even millions of examples, P-TTS leverages a small pool of only 90 manually selected reasoning instances and systematically varies exemplar augmentation through principled instruction prompting intensities at test time to synthesize diverse reasoning trajectory contexts. Then we finetune the various sizes of Qwen-2.5 models on P-TTS data. Across a suite of mathematical reasoning AIME2024 & 25, MATH500, and GPQA-Diamond, our P-TTS-7B and 32B models outperform the prior competitive baselines like S1 and S1.1 (1K-shot), achieving absolute accuracy gains of +26.66% and +30.00% on AIME'24 (7B), and +13.34% and +6.67% on AIME'25 (7B); P-TTS-32B yields gains of +23.33% and +16.63% on AIME'24, and +26.63% and +3.33% on AIME'25 (vs. S1 and S1.1, respectively), with comparable or better performance on MATH500 and GPQA-Diamond. We further show that P-TTS enhances zero-shot generalization accuracy on out-of-domain reasoning benchmarks of Gaokao, Kaoyan, OlympiadBench, AMC23, GradeSchoolMath, and Minerva. Our analysis suggests that test-time scaling effectively explores the latent space of reasoning patterns, amplifying LLM problem-solving with minimal annotation overhead, and further unlocking the reasoning potential and capabilities of LLMs. Prompting Test-Time Scaling offers a practical, low-cost way to elicit LLM reasoning in resource-constrained or rapidly evolving domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01171v3">Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 82 pages, 28 figures, 32 tables. Code is available at https://github.com/CHATS-lab/verbalize-sampling
    </div>
    <details class="paper-abstract">
      Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities"). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07651v1">OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with extended context windows enable powerful downstream applications but impose significant memory overhead, as caching all key-value (KV) states scales linearly with sequence length and batch size. Existing cache eviction methods address this by exploiting attention sparsity, yet they typically rank tokens heuristically using accumulated attention weights without considering their true impact on attention outputs. We propose Optimal Brain Cache (OBCache), a principled framework that formulates cache eviction as a layer-wise structured pruning problem. Building upon the Optimal Brain Damage (OBD) theory, OBCache quantifies token saliency by measuring the perturbation in attention outputs induced by pruning tokens, with closed-form scores derived for isolated keys, isolated values, and joint key-value pairs. Our scores account not only for attention weights but also for information from value states and attention outputs, thereby enhancing existing eviction strategies with output-aware signals. Experiments on LLaMA and Qwen models demonstrate that replacing the heuristic scores in existing works, which estimate token saliency across different query positions, with OBCache's output-aware scores consistently improves long-context accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v3">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 NeurIPS 2025 Track on Datasets and Benchmarks
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. Our project page is at \href{https://yifei-he.github.io/mergebench/}{https://yifei-he.github.io/mergebench/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06250v2">Scalable multilingual PII annotation for responsible AI in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) gain wider adoption, ensuring their reliable handling of Personally Identifiable Information (PII) across diverse regulatory contexts has become essential. This work introduces a scalable multilingual data curation framework designed for high-quality PII annotation across 13 underrepresented locales, covering approximately 336 locale-specific PII types. Our phased, human-in-the-loop annotation methodology combines linguistic expertise with rigorous quality assurance, leading to substantial improvements in recall and false positive rates from pilot, training, and production phases. By leveraging inter-annotator agreement metrics and root-cause analysis, the framework systematically uncovers and resolves annotation inconsistencies, resulting in high-fidelity datasets suitable for supervised LLM fine-tuning. Beyond reporting empirical gains, we highlight common annotator challenges in multilingual PII labeling and demonstrate how iterative, analytics-driven pipelines can enhance both annotation quality and downstream model reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17410v3">COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 23 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across various domains, yet their optimization remains a significant challenge due to the complex and high-dimensional loss landscapes they inhabit. While adaptive optimizers such as AdamW are widely used, they suffer from critical limitations, including an inability to capture interdependencies between coordinates and high memory consumption. Subsequent research, exemplified by SOAP, attempts to better capture coordinate interdependence but incurs greater memory overhead, limiting scalability for massive LLMs. An alternative approach aims to reduce memory consumption through low-dimensional projection, but this leads to substantial approximation errors, resulting in less effective optimization (e.g., in terms of per-token efficiency). In this paper, we propose COSMOS, a novel hybrid optimizer that leverages the varying importance of eigensubspaces in the gradient matrix to achieve memory efficiency without compromising optimization performance. The design of COSMOS is motivated by our empirical insights and practical considerations. Specifically, COSMOS applies SOAP to the leading eigensubspace, which captures the primary optimization dynamics, and MUON to the remaining eigensubspace, which is less critical but computationally expensive to handle with SOAP. This hybrid strategy significantly reduces memory consumption while maintaining robust optimization performance, making it particularly suitable for massive LLMs. Numerical experiments on various datasets and transformer architectures are provided to demonstrate the effectiveness of COSMOS. Our code is available at https://github.com/lliu606/COSMOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08813v1">The Model's Language Matters: A Comparative Privacy Analysis of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed across multilingual applications that handle sensitive data, yet their scale and linguistic variability introduce major privacy risks. Mostly evaluated for English, this paper investigates how language structure affects privacy leakage in LLMs trained on English, Spanish, French, and Italian medical corpora. We quantify six linguistic indicators and evaluate three attack vectors: extraction, counterfactual memorization, and membership inference. Results show that privacy vulnerability scales with linguistic redundancy and tokenization granularity: Italian exhibits the strongest leakage, while English shows higher membership separability. In contrast, French and Spanish display greater resilience due to higher morphological complexity. Overall, our findings provide the first quantitative evidence that language matters in privacy leakage, underscoring the need for language-aware privacy-preserving mechanisms in LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08779v1">Guiding Exploration in Reinforcement Learning Through LLM-Augmented Observations</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted to LM4Plan Workshop @ ICAPS 2025 (withdrawn before presentation due to lack of travel funding)
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) agents often struggle in sparse-reward environments where traditional exploration strategies fail to discover effective action sequences. Large Language Models (LLMs) possess procedural knowledge and reasoning capabilities from text pretraining that could guide RL exploration, but existing approaches create rigid dependencies where RL policies must follow LLM suggestions or incorporate them directly into reward functions. We propose a framework that provides LLM-generated action recommendations through augmented observation spaces, allowing RL agents to learn when to follow or ignore this guidance. Our method leverages LLMs' world knowledge and reasoning abilities while maintaining flexibility through soft constraints. We evaluate our approach on three BabyAI environments of increasing complexity and show that the benefits of LLM guidance scale with task difficulty. In the most challenging environment, we achieve 71% relative improvement in final success rates over baseline. The approach provides substantial sample efficiency gains, with agents reaching performance thresholds up to 9 times faster, and requires no modifications to existing RL algorithms. Our results demonstrate an effective method for leveraging LLM planning capabilities to accelerate RL training in challenging environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08776v1">Measuring Moral LLM Responses in Multilingual Capacities</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 10 pages, 5 figures; referenced articles: arXiv:2303.08774, arXiv:2303.12528, arXiv:2308.14132, arXiv:2505.12201, arXiv:2406.04428, arXiv:2407.02273, arXiv:2404.01268, arXiv:2502.09747, arXiv:2507.13474, arXiv:2505.21479, arXiv:2306.05685
    </div>
    <details class="paper-abstract">
      With LLM usage becoming widespread across countries, languages, and humanity more broadly, the need to understand and guardrail their multilingual responses increases. Large-scale datasets for testing and benchmarking have been created to evaluate and facilitate LLM responses across multiple dimensions. In this study, we evaluate the responses of frontier and leading open-source models in five dimensions across low and high-resource languages to measure LLM accuracy and consistency across multilingual contexts. We evaluate the responses using a five-point grading rubric and a judge LLM. Our study shows that GPT-5 performed the best on average in each category, while other models displayed more inconsistency across language and category. Most notably, in the Consent & Autonomy and Harm Prevention & Safety categories, GPT scored the highest with averages of 3.56 and 4.73, while Gemini 2.5 Pro scored the lowest with averages of 1.39 and 1.98, respectively. These findings emphasize the need for further testing on how linguistic shifts impact LLM responses across various categories and improvement in these areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08755v1">Robust Heuristic Algorithm Design with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      We posit that we can generate more robust and performant heuristics if we augment approaches using LLMs for heuristic design with tools that explain why heuristics underperform and suggestions about how to fix them. We find even simple ideas that (1) expose the LLM to instances where the heuristic underperforms; (2) explain why they occur; and (3) specialize design to regions in the input space, can produce more robust algorithms compared to existing techniques~ -- ~the heuristics we produce have a $\sim28\times$ better worst-case performance compared to FunSearch, improve average performance, and maintain the runtime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08741v1">Coordinates from Context: Using LLMs to Ground Complex Location References</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Under review at ARR
    </div>
    <details class="paper-abstract">
      Geocoding is the task of linking a location reference to an actual geographic location and is essential for many downstream analyses of unstructured text. In this paper, we explore the challenging setting of geocoding compositional location references. Building on recent work demonstrating LLMs' abilities to reason over geospatial data, we evaluate LLMs' geospatial knowledge versus reasoning skills relevant to our task. Based on these insights, we propose an LLM-based strategy for geocoding compositional location references. We show that our approach improves performance for the task and that a relatively small fine-tuned LLM can achieve comparable performance with much larger off-the-shelf models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06594v2">Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08710v1">Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 21 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Case-based reasoning is a cornerstone of U.S. legal practice, requiring professionals to argue about a current case by drawing analogies to and distinguishing from past precedents. While Large Language Models (LLMs) have shown remarkable capabilities, their proficiency in this complex, nuanced form of reasoning needs further investigation. We propose a formal framework that decomposes the process of identifying significant distinctions between cases into three-stage reasoning tasks. Our framework models cases using factual predicates called factors, organizes them into a legal knowledge hierarchy, and defines verifiable rules for identifying distinctions, analyzing their argumentative support, and evaluating their significance. Through comprehensive evaluation of modern reasoning LLMs, we reveal a paradox: while models achieve high accuracy on surface-level reasoning (Task 1), performance degrades on hierarchical reasoning (Task 2: 64.82%-92.09%) and collapses on integrated analysis (Task 3: 11.46%-33.99%). Most strikingly, we find that models consistently expend more computational resources on incorrect responses than correct ones, suggesting that "thinking longer" does not always mean "thinking smarter." Our work provides a methodology for fine-grained analysis of LLM reasoning capabilities in complex domains and reveals fundamental limitations that must be addressed for robust and trustworthy legal AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08705v1">ConPoSe: LLM-Guided Contact Point Selection for Scalable Cooperative Object Pushing</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Object transportation in cluttered environments is a fundamental task in various domains, including domestic service and warehouse logistics. In cooperative object transport, multiple robots must coordinate to move objects that are too large for a single robot. One transport strategy is pushing, which only requires simple robots. However, careful selection of robot-object contact points is necessary to push the object along a preplanned path. Although this selection can be solved analytically, the solution space grows combinatorially with the number of robots and object size, limiting scalability. Inspired by how humans rely on common-sense reasoning for cooperative transport, we propose combining the reasoning capabilities of Large Language Models with local search to select suitable contact points. Our LLM-guided local search method for contact point selection, ConPoSe, successfully selects contact points for a variety of shapes, including cuboids, cylinders, and T-shapes. We demonstrate that ConPoSe scales better with the number of robots and object size than the analytical approach, and also outperforms pure LLM-based selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08664v1">Faver: Boosting LLM-based RTL Generation with Function Abstracted Verifiable Middleware</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      LLM-based RTL generation is an interesting research direction, as it holds the potential to liberate the least automated stage in the current chip design. However, due to the substantial semantic gap between high-level specifications and RTL, coupled with limited training data, existing models struggle with generation accuracy. Drawing on human experience, design with verification helps improving accuracy. However, as the RTL testbench data are even more scarce, it is not friendly for LLMs. Although LLMs excel at higher-level languages like Python/C, they have a huge semantic gap from RTL. When implementing the same functionality, Python/C code and hardware code differ significantly in the spatiotemporal granularity, requiring the LLM not only to consider high-level functional semantics but also to ensure the low-level details align with the circuit code. It is not an easy task. In this paper, we propose a function abstracted verifiable middleware (Faver) that streamlines RTL verification in LLM-based workflows. By mixing LLM-friendly code structures with a rule-based template, Faver decouples the details of circuit verification, allowing the LLM to focus on the functionality itself. In our experiments on the SFT model and open-source models, Faver improved the model's generation accuracy by up to 14%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08663v1">A Novel Framework for Augmenting Rating Scale Tests with LLM-Scored Text Data</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Psychological assessments typically rely on structured rating scales, which cannot incorporate the rich nuance of a respondent's natural language. This study leverages recent LLM advances to harness qualitative data within a novel conceptual framework, combining LLM-scored text and traditional rating-scale items to create an augmented test. We demonstrate this approach using depression as a case study, developing and assessing the framework on a real-world sample of upper secondary students (n=693) and corresponding synthetic dataset (n=3,000). On held-out test sets, augmented tests achieved statistically significant improvements in measurement precision and accuracy. The information gain from the LLM items was equivalent to adding between 6.3 (real data) and 16.0 (synthetic data) items to the original 19-item test. Our approach marks a conceptual shift in automated scoring that bypasses its typical bottlenecks: instead of relying on pre-labelled data or complex expert-created rubrics, we empirically select the most informative LLM scoring instructions based on calculations of item information. This framework provides a scalable approach for leveraging the growing stream of transcribed text to enhance traditional psychometric measures, and we discuss its potential utility in clinical health and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08640v1">Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Android is the largest mobile platform, yet automatically building applications remains a practical challenge. While Large Language Models (LLMs) show promise for code repair, their use for fixing Android build errors remains underexplored. To address this gap, we first introduce AndroidBuildBench, a benchmark of 1,019 build failures curated from the commit histories of 43 open-source Android projects. Each problem is paired with a verified solution from a subsequent commit, ensuring that fixes are feasible. Second, we propose GradleFixer, an LLM agent with domain-specific tools for inspecting and manipulating the Gradle build environment. GradleFixer achieves a resolve rate of 81.4% (pass@1), significantly outperforming a state-of-the-art coding agent that relies on a general-purpose shell. GradleFixer's success suggests that while LLMs possess the high-level knowledge to solve these failures, they struggle to translate this knowledge into effective low-level actions using a general-purpose shell. We demonstrate the effectiveness of a strategy we term Tool Bridging, which replaces general-purpose shell commands with domain-aware abstractions. We hypothesize this approach works through two mechanisms: 1) it provides tools in an API-like format that LLMs use more reliably, and 2) it constrains the action space to relevant operations. This approach bridges the gap between the model's high-level reasoning and effective low-level execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08572v1">BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Scaling data and models has played a pivotal role in the remarkable progress of computer vision and language. Inspired by these domains, recent efforts in robotics have similarly focused on scaling both data and model size to develop more generalizable and robust policies. However, unlike vision and language, robotics lacks access to internet-scale demonstrations across diverse robotic tasks and environments. As a result, the scale of existing datasets typically suffers from the need for manual data collection and curation. To address this problem, here we propose BLAZER, a framework that learns manipulation policies from automatically generated training data. We build on the zero-shot capabilities of LLM planners and automatically generate demonstrations for diverse manipulation tasks in simulation. Successful examples are then used to finetune an LLM and to improve its planning capabilities without human supervision. Notably, while BLAZER training requires access to the simulator's state, we demonstrate direct transfer of acquired skills to sensor-based manipulation. Through extensive experiments, we show BLAZER to significantly improve zero-shot manipulation in both simulated and real environments. Moreover, BLAZER improves on tasks outside of its training pool and enables downscaling of LLM models. Our code and data will be made publicly available on the project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03438v3">BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have spurred growing interest in automatic theorem proving using Lean4, where effective tree search methods are crucial for navigating the underlying large proof search spaces. While the existing approaches primarily rely on value functions and/or Monte Carlo Tree Search (MCTS), the potential of simpler methods like Best-First Tree Search (BFS) remains underexplored. In this paper, we investigate whether BFS can achieve competitive performance in large-scale theorem proving tasks. We present BFS-Prover, a scalable expert iteration framework, featuring three key innovations. First, we implement strategic data filtering at each expert iteration round, excluding problems solvable via beam search node expansion to focus on harder cases. Second, we improve the sample efficiency of BFS through Direct Preference Optimization (DPO) applied to state-tactic pairs automatically annotated with compiler error feedback, refining the LLM's policy to prioritize productive expansions. Third, we employ length normalization in BFS to encourage exploration of deeper proof paths. BFS-Prover achieves a state-of-the-art score of $72.95\%$ on the MiniF2F test set and therefore challenges the perceived necessity of complex tree search methods, demonstrating that BFS can achieve competitive performance when properly scaled. To facilitate further research and development in this area, we have open-sourced our model at https://huggingface.co/ByteDance-Seed/BFS-Prover-V1-7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08544v1">SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained popularity in recent years, driving up the demand for inference. LLM inference is composed of two phases with distinct characteristics: a compute-bound prefill phase followed by a memory-bound decode phase. To efficiently serve LLMs, prior work proposes prefill-decode disaggregation to run each phase on separate hardware. However, existing hardware poorly matches the different requirements of each phase. Current datacenter GPUs and TPUs follow a more-is-better design philosophy that maximizes compute and memory resources, causing memory bandwidth underutilization in the prefill phase and compute underutilization in the decode phase. Such underutilization directly translates into increased serving costs. This paper proposes SPAD (Specialized Prefill and Decode hardware), adopting a less-is-more methodology to design specialized chips tailored to the distinct characteristics of prefill and decode phases. The proposed Prefill Chips have larger systolic arrays and use cost-effective GDDR memory, whereas the proposed Decode Chips retain high memory bandwidth but reduce compute capacity. Compared to modeled H100s, simulations show that the proposed Prefill Chips deliver 8% higher prefill performance on average at 52% lower hardware cost, while the proposed Decode Chips achieve 97% of the decode performance with 28% lower TDP. End-to-end simulations on production traces show that SPAD reduces hardware cost by 19%-41% and TDP by 2%-17% compared to modeled baseline clusters while offering the same performance. Even when models and workloads change, SPAD can reallocate either type of chip to run either phase and still achieve 11%-43% lower hardware costs, demonstrating the longevity of the SPAD design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08517v1">CaRT: Teaching LLM Agents to Know When They Know Enough</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Many tasks require learned models to strategically gather relevant information over multiple rounds of interaction before actually acting on a task. Strategic information gathering requires models to know not only how to effectively acquire information, but also when to stop gathering information and make a decision, in order to avoid overthinking or getting derailed when acting. In this paper, we formalize this problem and introduce Counterfactuals and Reasoning for Termination (CaRT), an approach for teaching LLMs when to stop seeking information. To appropriately learn when to terminate, CaRT fine-tunes LLMs using counterfactual pairs of trajectories, one where termination is appropriate and a minimally modified version of the same trajectory where it is not. It trains the LLM to explain the rationale for the termination decision in either case via verbal reasoning, and imbues this capability into the base LLM via fine-tuning. We instantiate CaRT in two domains: interactive medical diagnosis and math problem solving. In both domains, we find that CaRT improves the efficiency of information gathering and task success rate compared to other fine-tuning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06493v2">Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20600v4">Multi-Turn Human-LLM Interaction Through the Lens of a Two-Way Intelligibility Protocol</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Multi-Turn Interactions in Large Language Models (MTI-LLM) Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Our interest is in the design of software systems involving a human-expert interacting -- using natural language -- with a large language model (LLM) on data analysis tasks. For complex problems, it is possible that LLMs can harness human expertise and creativity to find solutions that were otherwise elusive. On one level, this interaction takes place through multiple turns of prompts from the human and responses from the LLM. Here we investigate a more structured approach based on an abstract protocol described in [3] for interaction between agents. The protocol is motivated by a notion of "two-way intelligibility" and is modelled by a pair of communicating finite-state machines. We provide an implementation of the protocol, and provide empirical evidence of using the implementation to mediate interactions between an LLM and a human-agent in two areas of scientific interest (radiology and drug design). We conduct controlled experiments with a human proxy (a database), and uncontrolled experiments with human subjects. The results provide evidence in support of the protocol's capability of capturing one- and two-way intelligibility in human-LLM interaction; and for the utility of two-way intelligibility in the design of human-machine systems. Our code is available at https://github.com/karannb/interact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18085v2">Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Original version uploaded on Sep 22, 2025. (v2): Extended Table 2 with additional analysis and referenced it in Sec 5.2
    </div>
    <details class="paper-abstract">
      Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by $\mathbf{2.8{-}3.1\times}$ while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to $\mathbf{7.9\times}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11194v3">Evaluating LLMs' Mathematical Reasoning in Financial Document Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 26 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), excel in natural language understanding, but their capability for complex mathematical reasoning with an amalgamation of structured tables and unstructured text is uncertain. This study explores LLMs' mathematical reasoning on four financial tabular question-answering datasets: TATQA, FinQA, ConvFinQA, and Multihiertt. Through extensive experiments with various models and prompting techniques, we assess how LLMs adapt to complex tables and mathematical tasks. We focus on sensitivity to table complexity and performance variations with an increasing number of arithmetic reasoning steps. The results provide insights into LLMs' capabilities and limitations in handling complex mathematical scenarios for semi-structured tables. Ultimately, we introduce a novel prompting technique tailored to semi-structured documents, matching or outperforming other baselines in performance while providing a nuanced understanding of LLMs abilities for such a task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08481v1">Forecasting the Buzz: Enriching Hashtag Popularity Prediction with LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted to CIKM 2025
    </div>
    <details class="paper-abstract">
      Hashtag trends ignite campaigns, shift public opinion, and steer millions of dollars in advertising spend, yet forecasting which tag goes viral is elusive. Classical regressors digest surface features but ignore context, while large language models (LLMs) excel at contextual reasoning but misestimate numbers. We present BuzzProphet, a reasoning-augmented hashtag popularity prediction framework that (1) instructs an LLM to articulate a hashtag's topical virality, audience reach, and timing advantage; (2) utilizes these popularity-oriented rationales to enrich the input features; and (3) regresses on these inputs. To facilitate evaluation, we release HashView, a 7,532-hashtag benchmark curated from social media. Across diverse regressor-LLM combinations, BuzzProphet reduces RMSE by up to 2.8% and boosts correlation by 30% over baselines, while producing human-readable rationales. Results demonstrate that using LLMs as context reasoners rather than numeric predictors injects domain insight into tabular models, yielding an interpretable and deployable solution for social media trend forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14756v2">LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Bayesian optimization (BO) is a sequential decision-making tool widely used for optimizing expensive black-box functions. Recently, Large Language Models (LLMs) have shown remarkable adaptability in low-data regimes, making them promising tools for black-box optimization by leveraging contextual knowledge to propose high-quality query points. However, relying solely on LLMs as optimization agents introduces risks due to their lack of explicit surrogate modeling and calibrated uncertainty, as well as their inherently opaque internal mechanisms. This structural opacity makes it difficult to characterize or control the exploration-exploitation trade-off, ultimately undermining theoretical tractability and reliability. To address this, we propose LLINBO: LLM-in-the-Loop BO, a hybrid framework for BO that combines LLMs with statistical surrogate experts (e.g., Gaussian Processes (GP)). The core philosophy is to leverage contextual reasoning strengths of LLMs for early exploration, while relying on principled statistical models to guide efficient exploitation. Specifically, we introduce three mechanisms that enable this collaboration and establish their theoretical guarantees. We end the paper with a real-life proof-of-concept in the context of 3D printing. The code to reproduce the results can be found at https://github.com/UMDataScienceLab/LLM-in-the-Loop-BO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08439v1">xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 24 Pages, 4 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Modern LLM deployments confront a widening cost-performance spectrum: premium models deliver strong reasoning but are expensive, while lightweight models are economical yet brittle on complex tasks. Static escalation rules and keyword heuristics under-utilize this spectrum and fail to adapt across task types. We present xRouter, a tool-calling-based routing system in which a learned router can either answer directly or invoke one or more external models. The router is trained end-to-end with reinforcement learning using an explicit, cost-aware reward that encodes cost-performance trade-offs, eliminating the need for hand-engineered routing rules. Our implementation encompasses the full reinforcement learning framework, including reward and cost accounting, as well as the deployment and evaluation pipelines. Across diverse benchmarks, xRouter achieves strong cost-performance trade-offs (e.g., substantial cost reductions at comparable task completion rates), and provides empirical insights into what reliably helps learned routing and what does not, ranging from model trainability to the difficulty of eliciting sophisticated orchestration behaviors in small open models. We hope these findings and our open implementation will serve as a practical substrate for advancing learned, cost-aware LLM orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07231v2">Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Causal reasoning is fundamental for Large Language Models (LLMs) to understand genuine cause-and-effect relationships beyond pattern matching. Existing benchmarks suffer from critical limitations such as reliance on synthetic data and narrow domain coverage. We introduce a novel benchmark constructed from casually identified relationships extracted from top-tier economics and finance journals, drawing on rigorous methodologies including instrumental variables, difference-in-differences, and regression discontinuity designs. Our benchmark comprises 40,379 evaluation items covering five task types across domains such as health, environment, technology, law, and culture. Experimental results on eight state-of-the-art LLMs reveal substantial limitations, with the best model achieving only 57.6\% accuracy. Moreover, model scale does not consistently translate to superior performance, and even advanced reasoning models struggle with fundamental causal relationship identification. These findings underscore a critical gap between current LLM capabilities and demands of reliable causal reasoning in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17320v2">AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Understanding the internal representations of large language models (LLMs) remains a central challenge for interpretability research. Sparse autoencoders (SAEs) offer a promising solution by decomposing activations into interpretable features, but existing approaches rely on fixed sparsity constraints that fail to account for input complexity. We propose AdaptiveK SAE (Adaptive Top K Sparse Autoencoders), a novel framework that dynamically adjusts sparsity levels based on the semantic complexity of each input. Leveraging linear probes, we demonstrate that context complexity is linearly encoded in LLM representations, and we use this signal to guide feature allocation during training. Experiments across ten language models (from 70M to 14B parameters) demonstrate that this complexity-driven adaptation significantly outperforms fixed-sparsity approaches on reconstruction fidelity, explained variance, cosine similarity and interpretability metrics while eliminating the computational burden of extensive hyperparameter tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03336v3">Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20435v2">The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Existing interpretability methods for Large Language Models (LLMs) often fall short by focusing on linear directions or isolated features, overlooking the high-dimensional, nonlinear, and relational geometry within model representations. This study focuses on how adversarial inputs systematically affect the internal representation spaces of LLMs, a topic which remains poorly understood. We propose persistent homology (PH), a tool from topological data analysis, as a principled framework to characterize the multi-scale dynamics within LLM activations. Using PH, we systematically analyze six state-of-the-art models under two distinct adversarial conditions, indirect prompt injection and backdoor fine-tuning, and identify a consistent topological signature of adversarial influence. Across architectures and model sizes, adversarial inputs induce ``topological compression'', where the latent space becomes structurally simpler, collapsing from varied, compact, small-scale features into fewer, dominant, and more dispersed large-scale ones. This topological signature is statistically robust across layers, highly discriminative, and provides interpretable insights into how adversarial effects emerge and propagate. By quantifying the shape of activations and neuronal information flow, our architecture-agnostic framework reveals fundamental invariants of representational change, offering a complementary perspective to existing interpretability methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08338v1">LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      Consumer research costs companies billions annually yet suffers from panel biases and limited scale. Large language models (LLMs) offer an alternative by simulating synthetic consumers, but produce unrealistic response distributions when asked directly for numerical ratings. We present semantic similarity rating (SSR), a method that elicits textual responses from LLMs and maps these to Likert distributions using embedding similarity to reference statements. Testing on an extensive dataset comprising 57 personal care product surveys conducted by a leading corporation in that market (9,300 human responses), SSR achieves 90% of human test-retest reliability while maintaining realistic response distributions (KS similarity > 0.85). Additionally, these synthetic respondents provide rich qualitative feedback explaining their ratings. This framework enables scalable consumer research simulations while preserving traditional survey metrics and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20749v2">Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabilities such as reasoning, planning, and personalization. Existing works aim to enhance LLM capabilities via domain-specific adaptation, which require additional training on accessible model parameters, an infeasible option for black-box LLMs. To address this challenge, we introduce Matryoshka Pilot (M-Pilot), a lightweight white-box LLM controller that guides a large-scale black-box LLM generator by decomposing complex tasks into a series of intermediate outputs. Specifically, we consider the black-box LLM as an environment, with M-Pilot serving as a policy to provide intermediate guidance through prompts for driving the black-box LLM. M-Pilot is trained to pivot the outputs of the black-box LLM aligning with preferences during iterative interaction, which enables controllable multi-turn generation and self-improvement in optimizing intermediate guidance. Empirical evaluations on diverse tasks demonstrate that our method effectively enhances the capabilities of black-box LLMs in complex, long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16723v2">LLM Fingerprinting via Semantically Conditioned Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Most LLM fingerprinting methods teach the model to respond to a few fixed queries with predefined atypical responses (keys). This memorization often does not survive common deployment steps such as finetuning or quantization, and such keys can be easily detected and filtered from LLM responses, ultimately breaking the fingerprint. To overcome these limitations we introduce LLM fingerprinting via semantically conditioned watermarks, replacing fixed query sets with a broad semantic domain, and replacing brittle atypical keys with a statistical watermarking signal diffused throughout each response. After teaching the model to watermark its responses only to prompts from a predetermined domain e.g., French language, the model owner can use queries from that domain to reliably detect the fingerprint and verify ownership. As we confirm in our thorough experimental evaluation, our fingerprint is both stealthy and robust to all common deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08638v4">Examining Multilingual Embedding Models Cross-Lingually Through LLM-Generated Adversarial Examples</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 To appear in EMNLP2025 Findings
    </div>
    <details class="paper-abstract">
      The evaluation of cross-lingual semantic search models is often limited to existing datasets from tasks such as information retrieval and semantic textual similarity. We introduce Cross-Lingual Semantic Discrimination (CLSD), a lightweight evaluation task that requires only parallel sentences and a Large Language Model (LLM) to generate adversarial distractors. CLSD measures an embedding model's ability to rank the true parallel sentence above semantically misleading but lexically similar alternatives. As a case study, we construct CLSD datasets for German--French in the news domain. Our experiments show that models fine-tuned for retrieval tasks benefit from pivoting through English, whereas bitext mining models perform best in direct cross-lingual settings. A fine-grained similarity analysis further reveals that embedding models differ in their sensitivity to linguistic perturbations. We release our code and datasets under AGPL-3.0: https://github.com/impresso/cross_lingual_semantic_discrimination
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16567v3">Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Finetuning open-weight Large Language Models (LLMs) is standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets leads to predictable behaviors. In this paper, we demonstrate, for the first time, that an adversary can create compromised LLMs that are performant and benign, yet exhibit adversarial behaviors once finetuned by downstream users. To this end, we propose an attack, FAB (Finetuning-activated Adversarial Behaviors), which compromises an LLM via meta-learning techniques that simulate downstream finetuning, explicitly optimizing for the emergence of adversarial behaviors in the finetuned models. At the same time, the compromised LLM is regularized to retain general capabilities and to exhibit no adversarial behaviors prior to finetuning. As a result, when users finetune (e.g., instruction-tuning, distillation, DPO) the seemingly benign model on their own datasets, they unknowingly trigger its dormant adversarial behavior. We experimentally demonstrate the effectiveness of FAB across multiple LLMs and three commonly considered target behaviors: unsolicited advertising, jailbreakability, and over-refusal. We show that FAB-triggers are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler, post-training algorithm). Our findings challenge prevailing assumptions on the security of finetuning, revealing a critical attack vector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19033v2">SelfRACG: Enabling LLMs to Self-Express and Retrieve for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Tsinghua&Xiaohongshu
    </div>
    <details class="paper-abstract">
      Existing retrieval-augmented code generation (RACG) methods typically use an external retrieval module to fetch semantically similar code snippets used for generating subsequent fragments. However, even for consecutive code fragments, the content often diverges due to logical progression, resulting in a content gap. This gap undermines the performance of current RACG methods, as \textit{external} retrieval modules based on content matching fail to infer the specific information need of LLMs to generate the next code fragment. Therefore, we propose \textbf{SelfRACG}, a novel paradigm that enables large language models (LLMs) to \textbf{Self}-express their information needs to enhance \textbf{RACG}. Specifically, SelfRACG includes an information need expression module and a two-stage information need-guided training strategy, which encourages LLMs to express their information need. Extensive experiments demonstrate that SelfRACG can retrieve external knowledge that better aligns with the LLM's own information needs, resulting in superior generation performance compared to vanilla RACG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10063v3">Hallucination Detection in LLMs with Topological Divergence on Attention Graphs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments - including evaluation on question answering and summarization tasks - show that our approach achieves state-of-the-art or competitive results on several benchmarks while requiring minimal annotated data and computational resources. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08255v1">Opponent Shaping in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 29 pages, 15 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being deployed as autonomous agents in real-world environments. As these deployments scale, multi-agent interactions become inevitable, making it essential to understand strategic behavior in such systems. A central open question is whether LLM agents, like reinforcement learning agents, can shape the learning dynamics and influence the behavior of others through interaction alone. In this paper, we present the first investigation of opponent shaping (OS) with LLM-based agents. Existing OS algorithms cannot be directly applied to LLMs, as they require higher-order derivatives, face scalability constraints, or depend on architectural components that are absent in transformers. To address this gap, we introduce ShapeLLM, an adaptation of model-free OS methods tailored for transformer-based agents. Using ShapeLLM, we examine whether LLM agents can influence co-players' learning dynamics across diverse game-theoretic environments. We demonstrate that LLM agents can successfully guide opponents toward exploitable equilibria in competitive games (Iterated Prisoner's Dilemma, Matching Pennies, and Chicken) and promote coordination and improve collective welfare in cooperative games (Iterated Stag Hunt and a cooperative version of the Prisoner's Dilemma). Our findings show that LLM agents can both shape and be shaped through interaction, establishing opponent shaping as a key dimension of multi-agent LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08242v1">Simulating Teams with LLM Agents: Interactive 2D Environments for Studying Human-AI Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Enabling users to create their own simulations offers a powerful way to study team dynamics and performance. We introduce VirTLab, a system that allows researchers and practitioners to design interactive, customizable simulations of team dynamics with LLM-based agents situated in 2D spatial environments. Unlike prior frameworks that restrict scenarios to predefined or static tasks, our approach enables users to build scenarios, assign roles, and observe how agents coordinate, move, and adapt over time. By bridging team cognition behaviors with scalable agent-based modeling, our system provides a testbed for investigating how environments influence coordination, collaboration, and emergent team behaviors. We demonstrate its utility by aligning simulated outcomes with empirical evaluations and a user study, underscoring the importance of customizable environments for advancing research on multi-agent simulations. This work contributes to making simulations accessible to both technical and non-technical users, supporting the design, execution, and analysis of complex multi-agent experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08233v1">Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Diffusion large language models (dLLMs) are promising alternatives to autoregressive large language models (AR-LLMs), as they potentially allow higher inference throughput. Reinforcement learning (RL) is a crucial component for dLLMs to achieve comparable performance with AR-LLMs on important tasks, such as reasoning. However, RL algorithms that are well-suited for dLLMs' unique characteristics have yet to be developed. This paper proposes Distribution Matching Policy Optimization (DMPO), a principled and theoretically grounded RL fine-tuning method specifically designed to enhance the reasoning capabilities of dLLMs by matching the dLLM policy distribution to the optimal, reward-tilted one through cross-entropy optimization. We identify a key challenge in the implementation with a small training batch size and propose several effective solutions through a novel weight baseline subtraction technique. DMPO exhibits superior performance on multiple reasoning benchmarks without supervised fine-tuning, with an accuracy improvement of up to $42.9\%$ over previously SOTA baselines and $55.8\%$ over the base model, underscoring the effectiveness of the distribution matching framework. Our code is available at https://github.com/yuchen-zhu-zyc/DMPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08211v1">LLMs Learn to Deceive Unintentionally: Emergent Misalignment in Dishonesty from Misaligned Samples to Biased Human-AI Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Previous research has shown that LLMs finetuned on malicious or incorrect completions within narrow domains (e.g., insecure code or incorrect medical advice) can become broadly misaligned to exhibit harmful behaviors, which is called emergent misalignment. In this work, we investigate whether this phenomenon can extend beyond safety behaviors to a broader spectrum of dishonesty and deception under high-stakes scenarios (e.g., lying under pressure and deceptive behavior). To explore this, we finetune open-sourced LLMs on misaligned completions across diverse domains. Experimental results demonstrate that LLMs show broadly misaligned behavior in dishonesty. Additionally, we further explore this phenomenon in a downstream combined finetuning setting, and find that introducing as little as 1% of misalignment data into a standard downstream task is sufficient to decrease honest behavior over 20%. Furthermore, we consider a more practical human-AI interaction environment where we simulate both benign and biased users to interact with the assistant LLM. Notably, we find that the assistant can be misaligned unintentionally to exacerbate its dishonesty with only 10% biased user population. In summary, we extend the study of emergent misalignment to the domain of dishonesty and deception under high-stakes scenarios, and demonstrate that this risk arises not only through direct finetuning, but also in downstream mixture tasks and practical human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08188v1">MetricalARGS: A Taxonomy for Studying Metrical Poetry with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Prior NLP work studying poetry has focused primarily on automatic poem generation and summarization. Many languages have well-studied traditions of poetic meter which enforce constraints on a poem in terms of syllable and phoneme patterns. Such advanced literary forms offer opportunities for probing deeper reasoning and language understanding in Large Language Models (LLMs) and their ability to follow strict pre-requisites and rules. In this paper, we introduce MetricalARGS, the first taxonomy of poetry-related NLP tasks designed to evaluate LLMs on metrical poetry across four dimensions: Analysis, Retrieval, Generation, and Support. We discuss how these tasks relate to existing NLP tasks, addressing questions around datasets and evaluation metrics. Taking Telugu as our example language, we illustrate how the taxonomy can be used in practice. MetricalARGS highlights the broader possibilities for understanding the capabilities and limitations of today's LLMs through the lens of metrical poetry.
    </details>
</div>
