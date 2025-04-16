# llm - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13543v2">BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Vision Language Models (VLMs) possess extensive knowledge and exhibit promising reasoning abilities, however, they still struggle to perform well in complex, dynamic environments. Real-world tasks require handling intricate interactions, advanced spatial reasoning, long-term planning, and continuous exploration of new strategies-areas in which we lack effective methodologies for comprehensively evaluating these capabilities. To address this gap, we introduce BALROG, a novel benchmark designed to assess the agentic capabilities of LLMs and VLMs through a diverse set of challenging games. Our benchmark incorporates a range of existing reinforcement learning environments with varying levels of difficulty, including tasks that are solvable by non-expert humans in seconds to extremely challenging ones that may take years to master (e.g., the NetHack Learning Environment). We devise fine-grained metrics to measure performance and conduct an extensive evaluation of several popular open-source and closed-source LLMs and VLMs. Our findings indicate that while current models achieve partial success in the easier games, they struggle significantly with more challenging tasks. Notably, we observe severe deficiencies in vision-based decision-making, as several models perform worse when visual representations of the environments are provided. We release BALROG as an open and user-friendly benchmark to facilitate future research and development in the agentic community. Code and Leaderboard at balrogai.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05447v2">TOBUGraph: Knowledge Graph-Based Retrieval for Enhanced LLM Performance Beyond RAG</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) is one of the leading and most widely used techniques for enhancing LLM retrieval capabilities, but it still faces significant limitations in commercial use cases. RAG primarily relies on the query-chunk text-to-text similarity in the embedding space for retrieval and can fail to capture deeper semantic relationships across chunks, is highly sensitive to chunking strategies, and is prone to hallucinations. To address these challenges, we propose TOBUGraph, a graph-based retrieval framework that first constructs the knowledge graph from unstructured data dynamically and automatically. Using LLMs, TOBUGraph extracts structured knowledge and diverse relationships among data, going beyond RAG's text-to-text similarity. Retrieval is achieved through graph traversal, leveraging the extracted relationships and structures to enhance retrieval accuracy, eliminating the need for chunking configurations while reducing hallucination. We demonstrate TOBUGraph's effectiveness in TOBU, a real-world application in production for personal memory organization and retrieval. Our evaluation using real user data demonstrates that TOBUGraph outperforms multiple RAG implementations in both precision and recall, significantly improving user experience through improved retrieval accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06621v4">Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 ICLR 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong reasoning and memorization capabilities via pretraining on massive textual corpora. However, this poses risk of privacy and copyright violations, highlighting the need for efficient machine unlearning methods that remove sensitive data without retraining from scratch. While Gradient Ascent (GA) is commonly used to unlearn by reducing the likelihood of generating unwanted content, it leads to unstable optimization and catastrophic forgetting of retrained knowledge. We find that combining GA with low-rank adaptation results in poor trade-offs between computational cost and generative performance. To address these challenges, we propose Low-rank Knowledge Unlearning (LoKU), a novel framework that enables robust and efficient unlearning for LLMs. First, we introduce Inverted Hinge Loss, which suppresses unwanted tokens while maintaining fluency by boosting the probability of the next most likely token. Second, we develop a data-adaptive initialization for LoRA adapters via low-rank approximation weighted with relative Fisher information, thereby focusing updates on parameters critical for removing targeted knowledge. Experiments on the Training Data Extraction Challenge dataset using GPT-Neo models as well as on the TOFU benchmark with Phi-1.5B and Llama2-7B models demonstrate that our approach effectively removes sensitive information while maintaining reasoning and generative capabilities with minimal impact. Our implementation can be found in https://github.com/csm9493/efficient-llm-unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09078v5">Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency. Code will be available at https://github.com/iamhankai/Forest-of-Thought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22968v2">HRET: A Self-Evolving LLM Evaluation Toolkit for Korean</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Recent advancements in Korean large language models (LLMs) have spurred numerous benchmarks and evaluation methodologies, yet the lack of a standardized evaluation framework has led to inconsistent results and limited comparability. To address this, we introduce HRET Haerae Evaluation Toolkit, an open-source, self-evolving evaluation framework tailored specifically for Korean LLMs. HRET unifies diverse evaluation methods, including logit-based scoring, exact-match, language-inconsistency penalization, and LLM-as-a-Judge assessments. Its modular, registry-based architecture integrates major benchmarks (HAE-RAE Bench, KMMLU, KUDGE, HRM8K) and multiple inference backends (vLLM, HuggingFace, OpenAI-compatible endpoints). With automated pipelines for continuous evolution, HRET provides a robust foundation for reproducible, fair, and transparent Korean NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01503v3">A Highly Scalable LLM Clusters with Optical Interconnect</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      We propose \emph{LumosCore} to build high-bandwidth and large-scale data center networks for LLM jobs. By replacing the core-layer electrical packet switches by optical circuit switches, \emph{LumosCore} could achieves $2\times$ increase in bandwidth or $8\times$ increase in network size. We offer the detailed design of \emph{LumosCore} at both deployment stage and running stage. At deployment stage, we propose Interleaved Wiring, which is compatible with all possible logical topologies. At running stage, we design polynomial-time algorithms for GPU placement, logical topology generating and OCS reconfiguration to minimize network contention and reduce impact to scheduled jobs. We evaluate \emph{LumosCore} using both testbed experiments and large-scale simulation. Compared to traditional hybrid optical/electrical architectures, \emph{LumosCore} increases the end-to-end training throughput by up to 39.5\% on a 128-node testbed. Compared to the state-of-art Clos architectures, \emph{LumosCore} reduces the average job completion time by up to 34.1\% in a 16k simulation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19178v2">UQABench: Evaluating User Embedding for Prompting LLMs in Personalized Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages, 3 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve remarkable success in natural language processing (NLP). In practical scenarios like recommendations, as users increasingly seek personalized experiences, it becomes crucial to incorporate user interaction history into the context of LLMs to enhance personalization. However, from a practical utility perspective, user interactions' extensive length and noise present challenges when used directly as text prompts. A promising solution is to compress and distill interactions into compact embeddings, serving as soft prompts to assist LLMs in generating personalized responses. Although this approach brings efficiency, a critical concern emerges: Can user embeddings adequately capture valuable information and prompt LLMs? To address this concern, we propose \name, a benchmark designed to evaluate the effectiveness of user embeddings in prompting LLMs for personalization. We establish a fair and standardized evaluation process, encompassing pre-training, fine-tuning, and evaluation stages. To thoroughly evaluate user embeddings, we design three dimensions of tasks: sequence understanding, action prediction, and interest perception. These evaluation tasks cover the industry's demands in traditional recommendation tasks, such as improving prediction accuracy, and its aspirations for LLM-based methods, such as accurately understanding user interests and enhancing the user experience. We conduct extensive experiments on various state-of-the-art methods for modeling user embeddings. Additionally, we reveal the scaling laws of leveraging user embeddings to prompt LLMs. The benchmark is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20197v2">Enhancing the Robustness of LLM-Generated Code: Empirical Study and Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of code generated by large language models (LLMs) is crucial for real-world reliability. However, existing evaluations predominantly focus on correctness, often neglecting key robustness concerns such as missing input validation and insufficient error handling. In this paper, we present the first empirical study on the robustness of LLM-generated code. We introduce novel robustness metrics and analyze four state-of-the-art code LLMs, revealing that, on average, 43.1% of their generated code is less robust than human-written counterparts. Notably, over 90% of robustness deficiencies stem from missing conditional checks, with 70% of these omissions occurring in the first line of code. Additionally, in 69% of cases where a conditional statement is necessary but absent, the "if" token still ranks third or higher in the model's predicted token probabilities, indicating an implicit recognition of control structures. Building on these findings, we propose RobGen, a framework designed to enhance code robustness without requiring model retraining. RobGen leverages two model-agnostic techniques: RobGen-Adj, which dynamically adjusts token probabilities during decoding to encourage the inclusion of control structures, and RobGen-Ins, which improves generated code by inserting missing conditionals after generation. Experimental results demonstrate that RobGen reduces the proportion of less robust model-generated code by 20.0%, significantly enhancing code reliability across diverse tasks. As a lightweight and adaptable solution, RobGen effectively mitigates robustness challenges in LLM-generated code. All code and data are available at https://github.com/SYSUSELab/RobGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06994v2">Phaedrus: Predicting Dynamic Application Behavior with Lightweight Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Application profiling is an indispensable technique for many software development tasks, such as code and memory layout optimizations, where optimization decisions are tailored to specific program profiles. Unfortunately, modern applications codebases exhibit highly variant behavior across different inputs, creating challenges for conventional profiling approaches that rely on a single representative execution instance. In this paper, we propose \textbf{Phaedrus}, a new \textit{compiler-assisted deep learning framework} designed to predict dynamic program behaviors across varied execution instances, specifically focusing on dynamic function call prediction.Such predicted call sequences are then used for producing optimized code pertinent to a given input. Traditional profile-guided optimization methods struggle with the input-dependent variability of modern applications, where profiling on different inputs yields divergent application behaviors. To address this, Phaedrus proposes two new approaches: \textit{Application Behavior Synthesis}, a profile-less approach where Large Language Models (LLMs) directly infer dynamic functions based on source code \& static compiler analysis, bypassing the need for traditional profiling, and \textit{Application Profile Generalization}, which uses generative models trained on compressed and augmented \textit{Whole Program Path} (WPP) based function profiles to predict application behavior under unseen inputs. Our experiments show that \textit{Phaedrus} can achieve upto $10^7X$ reduction in WPP function profile sizes, can predict most frequently executed functions that cover upto 85-99\% of the execution time, along with an average of 13.68\% (upto 65\%) reduction in application binary size, and an average of 2.8\% performance improvement over the traditional profile-guided optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06559v2">Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 22 pages, 11 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Language agents based on large language models (LLMs) have demonstrated great promise in automating web-based tasks. Recent work has shown that incorporating advanced planning algorithms, e.g., tree search, is advantageous over reactive planning for web agents. However, unlike simulated sandbox environments, real-world environments such as the web are rife with irreversible actions. This undermines the feasibility of backtracking, a cornerstone of (tree) search. Overly relying on test-time search also hurts efficiency. We advocate model-based planning for web agents that employs a world model to simulate and deliberate over the outcome of each candidate action before committing to one. We systematically explore this paradigm by (1) Proposing a model-based planning framework, WebDreamer, which employs LLMs to serve as both world models and value functions; (2) Training specialized LLMs as world models with a scalable data synthesis pipeline. Empirical results demonstrate that WebDreamer achieves substantial performance improvements over reactive baselines. It is competitive, while being 4-5 times more efficient, with tree search in sandbox environments (VisualWebArena) and also works effectively on real-world websites (Online-Mind2Web and Mind2Web-Live). Furthermore, our trained world model, Dreamer-7B, performs comparable to GPT-4o, highlighting the potential of specialized world models for efficient and effective planning in complex web environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15426v2">CodingTeachLLM: Empowering LLM's Coding Ability via AST Prior Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      In this paper, we introduce CodingTeachLLM, a large language model (LLM) designed for coding teaching. Specially, we aim to enhance the coding ability of LLM and lead it to better teaching mode in education context. Thus, we propose an end-to-end prior-based three-phases supervised fine-tuned model, which is proved more competitive than traditional fine-tuning method. More specifically, our model realizes the structural disassembly and incremental guided output of educational knowledge. To this end, we robustify data classification of three types via a sampler and overlap estimation neural network, and inject the preprocessing datasets into pre-trained model in three batches for LORA fine-tuning. Then, we design a prior module couples system prompt, vector databases, and abstract syntax tree task segmentation. Finally, the compression method and regularization constraint are applied to the prior-based fine-tuned model, followed by text filter at the output end to obtain incremental guided results. Our model represents the first research effort to truly embody the tutor role with the features of abundant educational knowledge, step-by-step incremental guided outputs and non-disclosure of answers. Extensive experiments report that our model also achieves state-of-the-art in code abilities compared to open-source models, reaching an impressive 75.10% on the HumanEval (@pass 1) benchmark. Additionally, our model maintains strong conversational capabilities, with the 13B quantized version achieving scores of 56.34, 50.60, and 45.27 respectively on the MMLU, C-Eval, and AGIEval (5 shot) dialogue evaluation benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04667v4">Non-Determinism of "Deterministic" LLM Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      LLM (large language model) practitioners commonly notice that outputs can vary for the same inputs under settings expected to be deterministic. Yet the questions of how pervasive this is, and with what impact on results, have not to our knowledge been systematically investigated. We investigate non-determinism in five LLMs configured to be deterministic when applied to eight common tasks in across 10 runs, in both zero-shot and few-shot settings. We see accuracy variations up to 15% across naturally occurring runs with a gap of best possible performance to worst possible performance up to 70%. In fact, none of the LLMs consistently delivers repeatable accuracy across all tasks, much less identical output strings. Sharing preliminary results with insiders has revealed that non-determinism perhaps essential to the efficient use of compute resources via co-mingled data in input buffers so this issue is not going away anytime soon. To better quantify our observations, we introduce metrics focused on quantifying determinism, TARr@N for the total agreement rate at N runs over raw output, and TARa@N for total agreement rate of parsed-out answers. Our code and data are publicly available at http://github.com/REDACTED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01141v2">How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Chain-of-thought prompting has emerged as a powerful technique for enabling large language models (LLMs) to solve complex reasoning tasks. However, these reasoning chains can be verbose, raising concerns about efficiency. In response, recent works have sought to decrease response lengths through simple prompting strategies (e.g. 'be concise'). In this work, we conduct the first systematic study of the relationship between reasoning length and model performance across a diverse range of compression instructions (e.g. 'use 10 words or less' or 'remove all punctuation'). In doing so, we discover a universal tradeoff between reasoning length and accuracy that persists across even very distinct reasoning chains. We demonstrate that this tradeoff emerges from a sharp threshold behavior at the question level: each task has an intrinsic 'token complexity' - a minimal number of tokens required for successful problem-solving. We show how token complexity enables us to compute information-theoretic limits on the accuracy-compression tradeoff, and find that prompt-based compression strategies operate far from these theoretical limits. This suggests there may be significant room for improvement and our framework provides a benchmark to help researchers evaluate progress in reasoning efficiency. Our work also highlights the importance of adaptive compression -- giving shorter responses for easier questions -- and we show that token complexity is a useful tool for measuring this capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19326v2">Process or Result? Manipulated Ending Tokens Can Mislead Reasoning LLMs to Ignore the Correct Reasoning Steps</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Recent reasoning large language models (LLMs) have demonstrated remarkable improvements in mathematical reasoning capabilities through long Chain-of-Thought. The reasoning tokens of these models enable self-correction within reasoning chains, enhancing robustness. This motivates our exploration: how vulnerable are reasoning LLMs to subtle errors in their input reasoning chains? We introduce "Compromising Thought" (CPT), a vulnerability where models presented with reasoning tokens containing manipulated calculation results tend to ignore correct reasoning steps and adopt incorrect results instead. Through systematic evaluation across multiple reasoning LLMs, we design three increasingly explicit prompting methods to measure CPT resistance, revealing that models struggle significantly to identify and correct these manipulations. Notably, contrary to existing research suggesting structural alterations affect model performance more than content modifications, we find that local ending token manipulations have greater impact on reasoning outcomes than structural changes. Moreover, we discover a security vulnerability in DeepSeek-R1 where tampered reasoning tokens can trigger complete reasoning cessation. Our work enhances understanding of reasoning robustness and highlights security considerations for reasoning-intensive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12836v2">EditRoom: LLM-parameterized Graph Diffusion for Composable 3D Room Layout Editing</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Given the steep learning curve of professional 3D software and the time-consuming process of managing large 3D assets, language-guided 3D scene editing has significant potential in fields such as virtual reality, augmented reality, and gaming. However, recent approaches to language-guided 3D scene editing either require manual interventions or focus only on appearance modifications without supporting comprehensive scene layout changes. In response, we propose EditRoom, a unified framework capable of executing a variety of layout edits through natural language commands, without requiring manual intervention. Specifically, EditRoom leverages Large Language Models (LLMs) for command planning and generates target scenes using a diffusion-based method, enabling six types of edits: rotate, translate, scale, replace, add, and remove. To address the lack of data for language-guided 3D scene editing, we have developed an automatic pipeline to augment existing 3D scene synthesis datasets and introduced EditRoom-DB, a large-scale dataset with 83k editing pairs, for training and evaluation. Our experiments demonstrate that our approach consistently outperforms other baselines across all metrics, indicating higher accuracy and coherence in language-guided scene layout editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09577v2">Flash normalization: fast normalization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      RMSNorm is used by many LLMs such as Llama, Mistral, and OpenELM. This paper details FlashNorm, which is an exact but faster implementation of RMSNorm followed by linear layers. FlashNorm also speeds up Layer Normalization and its recently proposed replacement Dynamic Tanh (DyT) arXiv:2503.10622. See https://github.com/OpenMachine-ai/transformer-tricks for code and more transformer tricks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01241v1">Catastrophic Forgetting in LLMs: A Comparative Analysis Across Language Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced Natural Language Processing (NLP), particularly in Natural Language Understanding (NLU) tasks. As we progress toward an agentic world where LLM-based agents autonomously handle specialized tasks, it becomes crucial for these models to adapt to new tasks without forgetting previously learned information - a challenge known as catastrophic forgetting. This study evaluates the continual fine-tuning of various open-source LLMs with different parameter sizes (specifically models under 10 billion parameters) on key NLU tasks from the GLUE benchmark, including SST-2, MRPC, CoLA, and MNLI. By employing prompt engineering and task-specific adjustments, we assess and compare the models' abilities to retain prior knowledge while learning new tasks. Our results indicate that models such as Phi-3.5-mini exhibit minimal forgetting while maintaining strong learning capabilities, making them well-suited for continual learning environments. Additionally, models like Orca-2-7b and Qwen2.5-7B demonstrate impressive learning abilities and overall performance after fine-tuning. This work contributes to understanding catastrophic forgetting in LLMs and highlights prompting engineering to optimize model performance for continual learning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15487v2">Multi-LLM Text Summarization</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      In this work, we propose a Multi-LLM summarization framework, and investigate two different multi-LLM strategies including centralized and decentralized. Our multi-LLM summarization framework has two fundamentally important steps at each round of conversation: generation and evaluation. These steps are different depending on whether our multi-LLM decentralized summarization is used or centralized. In both our multi-LLM decentralized and centralized strategies, we have k different LLMs that generate diverse summaries of the text. However, during evaluation, our multi-LLM centralized summarization approach leverages a single LLM to evaluate the summaries and select the best one whereas k LLMs are used for decentralized multi-LLM summarization. Overall, we find that our multi-LLM summarization approaches significantly outperform the baselines that leverage only a single LLM by up to 3x. These results indicate the effectiveness of multi-LLM approaches for summarization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01234v1">First Field-Trial Demonstration of L4 Autonomous Optical Network for Distributed AI Training Communication: An LLM-Powered Multi-AI-Agent Solution</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Submitted to the PDP session of the Optical Fiber Communications Conference (OFC) 2025
    </div>
    <details class="paper-abstract">
      We demonstrate the first cross-domain cross-layer level-4 autonomous optical network via a multi-AI-agent system. Field trials show 98 percent task completion rate across the distributed AI training lifecycle-3.2x higher than single agents using state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01205v1">Epistemic Alignment: A Mediating Framework for User-LLM Knowledge Delivery</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      LLMs increasingly serve as tools for knowledge acquisition, yet users cannot effectively specify how they want information presented. When users request that LLMs "cite reputable sources," "express appropriate uncertainty," or "include multiple perspectives," they discover that current interfaces provide no structured way to articulate these preferences. The result is prompt sharing folklore: community-specific copied prompts passed through trust relationships rather than based on measured efficacy. We propose the Epistemic Alignment Framework, a set of ten challenges in knowledge transmission derived from the philosophical literature of epistemology, concerning issues such as evidence quality assessment and calibration of testimonial reliance. The framework serves as a structured intermediary between user needs and system capabilities, creating a common vocabulary to bridge the gap between what users want and what systems deliver. Through a thematic analysis of custom prompts and personalization strategies shared on online communities where these issues are actively discussed, we find users develop elaborate workarounds to address each of the challenges. We then apply our framework to two prominent model providers, OpenAI and Anthropic, through content analysis of their documented policies and product features. Our analysis shows that while these providers have partially addressed the challenges we identified, they fail to establish adequate mechanisms for specifying epistemic preferences, lack transparency about how preferences are implemented, and offer no verification tools to confirm whether preferences were followed. For AI developers, the Epistemic Alignment Framework offers concrete guidance for supporting diverse approaches to knowledge; for users, it works toward information delivery that aligns with their specific needs rather than defaulting to one-size-fits-all approaches.
    </details>
</div>
