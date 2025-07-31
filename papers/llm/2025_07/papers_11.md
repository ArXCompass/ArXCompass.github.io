# llm - 2025_07

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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- Part 11

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03014v1">Intrinsic Fingerprint of LLMs: Continue Training is NOT All You Need to Steal A Model!</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 This paper flags a potential case of model plagiarism, copyright violation, and information fabrication in arXiv:2505.21411
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face significant copyright and intellectual property challenges as the cost of training increases and model reuse becomes prevalent. While watermarking techniques have been proposed to protect model ownership, they may not be robust to continue training and development, posing serious threats to model attribution and copyright protection. This work introduces a simple yet effective approach for robust LLM fingerprinting based on intrinsic model characteristics. We discover that the standard deviation distributions of attention parameter matrices across different layers exhibit distinctive patterns that remain stable even after extensive continued training. These parameter distribution signatures serve as robust fingerprints that can reliably identify model lineage and detect potential copyright infringement. Our experimental validation across multiple model families demonstrates the effectiveness of our method for model authentication. Notably, our investigation uncovers evidence that a recently Pangu Pro MoE model released by Huawei is derived from Qwen-2.5 14B model through upcycling techniques rather than training from scratch, highlighting potential cases of model plagiarism, copyright violation, and information fabrication. These findings underscore the critical importance of developing robust fingerprinting methods for protecting intellectual property in large-scale model development and emphasize that deliberate continued training alone is insufficient to completely obscure model origins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03004v1">CLUES: Collaborative High-Quality Data Selection for LLMs via Training Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recent research has highlighted the importance of data quality in scaling large language models (LLMs). However, automated data quality control faces unique challenges in collaborative settings where sharing is not allowed directly between data silos. To tackle this issue, this paper proposes a novel data quality control technique based on the notion of data influence on the training dynamics of LLMs, that high quality data are more likely to have similar training dynamics to the anchor dataset. We then leverage the influence of the training dynamics to select high-quality data from different private domains, with centralized model updates on the server side in a collaborative training fashion by either model merging or federated learning. As for the data quality indicator, we compute the per-sample gradients with respect to the private data and the anchor dataset, and use the trace of the accumulated inner products as a measurement of data quality. In addition, we develop a quality control evaluation tailored for collaborative settings with heterogeneous domain data. Experiments show that training on the high-quality data selected by our method can often outperform other data selection methods for collaborative fine-tuning of LLMs, across diverse private domain datasets, in medical, multilingual and financial settings. Our code is released at github.com/Ryan0v0/CLUES.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03001v1">Evaluating Hierarchical Clinical Document Classification Using Reasoning-Based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      This study evaluates how well large language models (LLMs) can classify ICD-10 codes from hospital discharge summaries, a critical but error-prone task in healthcare. Using 1,500 summaries from the MIMIC-IV dataset and focusing on the 10 most frequent ICD-10 codes, the study tested 11 LLMs, including models with and without structured reasoning capabilities. Medical terms were extracted using a clinical NLP tool (cTAKES), and models were prompted in a consistent, coder-like format. None of the models achieved an F1 score above 57%, with performance dropping as code specificity increased. Reasoning-based models generally outperformed non-reasoning ones, with Gemini 2.5 Pro performing best overall. Some codes, such as those related to chronic heart disease, were classified more accurately than others. The findings suggest that while LLMs can assist human coders, they are not yet reliable enough for full automation. Future work should explore hybrid methods, domain-specific model training, and the use of structured clinical data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01936v1">The Thin Line Between Comprehension and Persuasion in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are excellent at maintaining high-level, convincing dialogues. They are being fast deployed as chatbots and evaluators in sensitive areas, such as peer review and mental health applications. This, along with the disparate accounts on their reasoning capabilities, calls for a closer examination of LLMs and their comprehension of dialogue. In this work we begin by evaluating LLMs' ability to maintain a debate--one of the purest yet most complex forms of human communication. Then we measure how this capability relates to their understanding of what is being talked about, namely, their comprehension of dialogical structures and the pragmatic context. We find that LLMs are capable of maintaining coherent, persuasive debates, often swaying the beliefs of participants and audiences alike. We also note that awareness or suspicion of AI involvement encourage people to be more critical of the arguments made. When polling LLMs on their comprehension of deeper structures of dialogue, however, they cannot demonstrate said understanding. Our findings tie the shortcomings of LLMs-as-evaluators to their (in)ability to understand the context. More broadly, for the field of argumentation theory we posit that, if an agent can convincingly maintain a dialogue, it is not necessary for it to know what it is talking about. Hence, the modelling of pragmatic context and coherence are secondary to effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03814v3">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01872v1">DIY-MKG: An LLM-Based Polyglot Language Learning System</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Submitted to EMNLP 2025 System Demonstration
    </div>
    <details class="paper-abstract">
      Existing language learning tools, even those powered by Large Language Models (LLMs), often lack support for polyglot learners to build linguistic connections across vocabularies in multiple languages, provide limited customization for individual learning paces or needs, and suffer from detrimental cognitive offloading. To address these limitations, we design Do-It-Yourself Multilingual Knowledge Graph (DIY-MKG), an open-source system that supports polyglot language learning. DIY-MKG allows the user to build personalized vocabulary knowledge graphs, which are constructed by selective expansion with related words suggested by an LLM. The system further enhances learning through rich annotation capabilities and an adaptive review module that leverages LLMs for dynamic, personalized quiz generation. In addition, DIY-MKG allows users to flag incorrect quiz questions, simultaneously increasing user engagement and providing a feedback loop for prompt refinement. Our evaluation of LLM-based components in DIY-MKG shows that vocabulary expansion is reliable and fair across multiple languages, and that the generated quizzes are highly accurate, validating the robustness of DIY-MKG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01844v1">Low-Perplexity LLM-Generated Sequences and Where To Find Them</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Camera-ready version. Accepted to ACL 2025. 10 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly widespread, understanding how specific training data shapes their outputs is crucial for transparency, accountability, privacy, and fairness. To explore how LLMs leverage and replicate their training data, we introduce a systematic approach centered on analyzing low-perplexity sequences - high-probability text spans generated by the model. Our pipeline reliably extracts such long sequences across diverse topics while avoiding degeneration, then traces them back to their sources in the training data. Surprisingly, we find that a substantial portion of these low-perplexity spans cannot be mapped to the corpus. For those that do match, we quantify the distribution of occurrences across source documents, highlighting the scope and nature of verbatim recall and paving a way toward better understanding of how LLMs training data impacts their behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01827v1">APRMCTS: Improving LLM-based Automated Program Repair with Iterative Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Automated Program Repair (APR) attempts to fix software bugs without human intervention, which plays a crucial role in software development and maintenance. Recently, with the advances in Large Language Models (LLMs), a rapidly increasing number of APR techniques have been proposed with remarkable performance. However, existing LLM-based APR techniques typically adopt trial-and-error strategies, which suffer from two major drawbacks: (1) inherently limited patch effectiveness due to local exploration, and (2) low search efficiency due to redundant exploration. In this paper, we propose APRMCTS, which uses iterative tree search to improve LLM-based APR. APRMCTS incorporates Monte Carlo Tree Search (MCTS) into patch searching by performing a global evaluation of the explored patches and selecting the most promising one for subsequent refinement and generation. APRMCTS effectively resolves the problems of falling into local optima and thus helps improve the efficiency of patch searching. Our experiments on 835 bugs from Defects4J demonstrate that, when integrated with GPT-3.5, APRMCTS can fix a total of 201 bugs, which outperforms all state-of-the-art baselines. Besides, APRMCTS helps GPT-4o-mini, GPT-3.5, Yi-Coder-9B, and Qwen2.5-Coder-7B to fix 30, 27, 37, and 28 more bugs, respectively. More importantly, APRMCTS boasts a significant performance advantage while employing small patch size (16 and 32), notably fewer than the 500 and 10,000 patches adopted in previous studies. In terms of cost, compared to existing state-of-the-art LLM-based APR methods, APRMCTS has time and monetary costs of less than 20% and 50%, respectively. Our extensive study demonstrates that APRMCTS exhibits good effectiveness and efficiency, with particular advantages in addressing complex bugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01806v1">LoRA Fine-Tuning Without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 5-page main paper (excluding references) + 11-page appendix, 3 tables, 1 figure. Accepted to ICML 2025 Workshop on Efficient Systems for Foundation Models
    </div>
    <details class="paper-abstract">
      Low-Rank Adapters (LoRAs) have transformed the fine-tuning of Large Language Models (LLMs) by enabling parameter-efficient updates. However, their widespread adoption remains limited by the reliance on GPU-based training. In this work, we propose a theoretically grounded approach to LoRA fine-tuning designed specifically for users with limited computational resources, particularly those restricted to standard laptop CPUs. Our method learns a meta-operator that maps any input dataset, represented as a probability distribution, to a set of LoRA weights by leveraging a large bank of pre-trained adapters for the Mistral-7B-Instruct-v0.2 model. Instead of performing new gradient-based updates, our pipeline constructs adapters via lightweight combinations of existing LoRAs directly on CPU. While the resulting adapters do not match the performance of GPU-trained counterparts, they consistently outperform the base Mistral model on downstream tasks, offering a practical and accessible alternative to traditional GPU-based fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00672v1">Toward Edge General Intelligence with Multiple-Large Language Model (Multi-LLM): Architecture, Trust, and Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Edge computing enables real-time data processing closer to its source, thus improving the latency and performance of edge-enabled AI applications. However, traditional AI models often fall short when dealing with complex, dynamic tasks that require advanced reasoning and multimodal data processing. This survey explores the integration of multi-LLMs (Large Language Models) to address this in edge computing, where multiple specialized LLMs collaborate to enhance task performance and adaptability in resource-constrained environments. We review the transition from conventional edge AI models to single LLM deployment and, ultimately, to multi-LLM systems. The survey discusses enabling technologies such as dynamic orchestration, resource scheduling, and cross-domain knowledge transfer that are key for multi-LLM implementation. A central focus is on trusted multi-LLM systems, ensuring robust decision-making in environments where reliability and privacy are crucial. We also present multimodal multi-LLM architectures, where multiple LLMs specialize in handling different data modalities, such as text, images, and audio, by integrating their outputs for comprehensive analysis. Finally, we highlight future directions, including improving resource efficiency, trustworthy governance multi-LLM systems, while addressing privacy, trust, and robustness concerns. This survey provides a valuable reference for researchers and practitioners aiming to leverage multi-LLM systems in edge computing applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02990v1">`For Argument's Sake, Show Me How to Harm Myself!': Jailbreaking LLMs in Suicide and Self-Harm Contexts</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have led to increasingly sophisticated safety protocols and features designed to prevent harmful, unethical, or unauthorized outputs. However, these guardrails remain susceptible to novel and creative forms of adversarial prompting, including manually generated test cases. In this work, we present two new test cases in mental health for (i) suicide and (ii) self-harm, using multi-step, prompt-level jailbreaking and bypass built-in content and safety filters. We show that user intent is disregarded, leading to the generation of detailed harmful content and instructions that could cause real-world harm. We conduct an empirical evaluation across six widely available LLMs, demonstrating the generalizability and reliability of the bypass. We assess these findings and the multilayered ethical tensions that they present for their implications on prompt-response filtering and context- and task-specific model development. We recommend a more comprehensive and systematic approach to AI safety and ethics while emphasizing the need for continuous adversarial testing in safety-critical AI deployments. We also argue that while certain clearly defined safety measures and guardrails can and must be implemented in LLMs, ensuring robust and comprehensive safety across all use cases and domains remains extremely challenging given the current technical maturity of general-purpose LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00657v1">Generative Exaggeration in LLM Social Agents: Consistency, Bias, and Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      We investigate how Large Language Models (LLMs) behave when simulating political discourse on social media. Leveraging 21 million interactions on X during the 2024 U.S. presidential election, we construct LLM agents based on 1,186 real users, prompting them to reply to politically salient tweets under controlled conditions. Agents are initialized either with minimal ideological cues (Zero Shot) or recent tweet history (Few Shot), allowing one-to-one comparisons with human replies. We evaluate three model families (Gemini, Mistral, and DeepSeek) across linguistic style, ideological consistency, and toxicity. We find that richer contextualization improves internal consistency but also amplifies polarization, stylized signals, and harmful language. We observe an emergent distortion that we call "generation exaggeration": a systematic amplification of salient traits beyond empirical baselines. Our analysis shows that LLMs do not emulate users, they reconstruct them. Their outputs, indeed, reflect internal optimization dynamics more than observed behavior, introducing structural biases that compromise their reliability as social proxies. This challenges their use in content moderation, deliberative simulations, and policy modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00543v1">Reliable Annotations with Less Effort: Evaluating LLM-Human Collaboration in Search Clarifications</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 9 pages,5 figures
    </div>
    <details class="paper-abstract">
      Despite growing interest in using large language models (LLMs) to automate annotation, their effectiveness in complex, nuanced, and multi-dimensional labelling tasks remains relatively underexplored. This study focuses on annotation for the search clarification task, leveraging a high-quality, multi-dimensional dataset that includes five distinct fine-grained annotation subtasks. Although LLMs have shown impressive capabilities in general settings, our study reveals that even state-of-the-art models struggle to replicate human-level performance in subjective or fine-grained evaluation tasks. Through a systematic assessment, we demonstrate that LLM predictions are often inconsistent, poorly calibrated, and highly sensitive to prompt variations. To address these limitations, we propose a simple yet effective human-in-the-loop (HITL) workflow that uses confidence thresholds and inter-model disagreement to selectively involve human review. Our findings show that this lightweight intervention significantly improves annotation reliability while reducing human effort by up to 45%, offering a relatively scalable and cost-effective yet accurate path forward for deploying LLMs in real-world evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00507v1">LLM-Mesh: Enabling Elastic Sharing for Serverless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      The rise of LLMs has driven demand for private serverless deployments, characterized by moderate-scale models and infrequent requests. While existing solutions follow exclusive GPU deployment, we take a step back to explore modern platforms and find that: Emerging CPU architectures with built-in accelerators are capable of serving LLMs but remain underutilized, and both CPUs and GPUs can accommodate multiple LLMs simultaneously. We propose LLM-Mesh, a serverless inference scheme for small-to-mid-sized LLMs that enables elastic sharing across heterogeneous hardware. LLM-Mesh tackles three fundamental challenges: (1) precise, fine-grained compute resource allocation at token-level to handle fluctuating computational demands; (2) a coordinated and forward-looking memory scaling mechanism to detect out-of-memory hazards and reduce operational overhead; and (3) a dual approach that reduces resource fragmentation through proactive preemption and reactive bin-packing. Experimental results on 4 32-core CPUs and 4 A100 GPUs show that LLM-Meshimproves service capacity by 44% - 63% through sharing, while further leveraging CPUs boosts this to 91% - 159%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00439v1">Beyond Sociodemographic Prompting: Using Supervision to Align LLMs with Human Response Distributions</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      The ability to accurately predict how different population groups would answer subjective questions would have great value. In this work, we show that use of relatively simple supervision can greatly improve language model alignment with diverse population groups, as measured over three datasets spanning various topics. Beyond evaluating average performance, we also report how alignment varies across specific groups. The simplicity and generality of our approach promotes easy adoption, while our broad findings provide useful guidance for when to use or not use our approach in practice. By conducting evaluation over many LLMs and prompting strategies, along with open-sourcing our work, we provide a useful benchmark to stimulate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00432v1">Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00418v1">Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and High-Performance GPUs</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 To appear in Proceedings of the Practice and Experience in Advanced Research Computing (PEARC '25)
    </div>
    <details class="paper-abstract">
      This study presents a benchmarking analysis of the Qualcomm Cloud AI 100 Ultra (QAic) accelerator for large language model (LLM) inference, evaluating its energy efficiency (throughput per watt) and performance against leading NVIDIA (A100, H200) and AMD (MI300A) GPUs within the National Research Platform (NRP) ecosystem. A total of 15 open-source LLMs, ranging from 117 million to 90 billion parameters, are served using the vLLM framework. The QAic inference cards appears to be energy efficient and performs well in the energy efficiency metric in most cases. The findings offer insights into the potential of the Qualcomm Cloud AI 100 Ultra for high-performance computing (HPC) applications within the National Research Platform (NRP).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00406v1">Partnering with AI: A Pedagogical Feedback System for LLM Integration into Programming Education</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 This is an extended version of a poster paper accepted and published at ECTEL-2025
    </div>
    <details class="paper-abstract">
      Feedback is one of the most crucial components to facilitate effective learning. With the rise of large language models (LLMs) in recent years, research in programming education has increasingly focused on automated feedback generation to help teachers provide timely support to every student. However, prior studies often overlook key pedagogical principles, such as mastery and progress adaptation, that shape effective feedback strategies. This paper introduces a novel pedagogical framework for LLM-driven feedback generation derived from established feedback models and local insights from secondary school teachers. To evaluate this framework, we implemented a web-based application for Python programming with LLM-based feedback that follows the framework and conducted a mixed-method evaluation with eight secondary-school computer science teachers. Our findings suggest that teachers consider that, when aligned with the framework, LLMs can effectively support students and even outperform human teachers in certain scenarios through instant and precise feedback. However, we also found several limitations, such as its inability to adapt feedback to dynamic classroom contexts. Such a limitation highlights the need to complement LLM-generated feedback with human expertise to ensure effective student learning. This work demonstrates an effective way to use LLMs for feedback while adhering to pedagogical standards and highlights important considerations for future systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17336v2">Privacy-Preserving LLM Interaction with Socratic Chain-of-Thought Reasoning and Homomorphically Encrypted Vector Databases</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as personal agents, accessing sensitive user data such as calendars, emails, and medical records. Users currently face a trade-off: They can send private records, many of which are stored in remote databases, to powerful but untrusted LLM providers, increasing their exposure risk. Alternatively, they can run less powerful models locally on trusted devices. We bridge this gap. Our Socratic Chain-of-Thought Reasoning first sends a generic, non-private user query to a powerful, untrusted LLM, which generates a Chain-of-Thought (CoT) prompt and detailed sub-queries without accessing user data. Next, we embed these sub-queries and perform encrypted sub-second semantic search using our Homomorphically Encrypted Vector Database across one million entries of a single user's private data. This represents a realistic scale of personal documents, emails, and records accumulated over years of digital activity. Finally, we feed the CoT prompt and the decrypted records to a local language model and generate the final response. On the LoCoMo long-context QA benchmark, our hybrid framework, combining GPT-4o with a local Llama-3.2-1B model, outperforms using GPT-4o alone by up to 7.1 percentage points. This demonstrates a first step toward systems where tasks are decomposed and split between untrusted strong LLMs and weak local ones, preserving user privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11620v3">Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 18 pages and 3 References Pages
    </div>
    <details class="paper-abstract">
      In this work, we explore uncertainty estimation as a proxy for correctness in LLM-generated code. To this end, we adapt two state-of-the-art techniques from natural language generation -- one based on entropy and another on mutual information -- to the domain of code generation. Given the distinct semantic properties of code, we introduce modifications, including a semantic equivalence check based on symbolic execution. Our findings indicate a strong correlation between the uncertainty computed through these techniques and correctness, highlighting the potential of uncertainty estimation for quality assessment. Additionally, we propose a simplified version of the entropy-based method that assumes a uniform distribution over the LLM's responses, demonstrating comparable effectiveness. Using these techniques, we develop an abstention policy that prevents the model from making predictions when uncertainty is high, reducing incorrect outputs to near zero. Our evaluation on the LiveCodeBench shows that our approach significantly outperforms a baseline relying solely on LLM-reported log-probabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17080v2">Not Minds, but Signs: Reframing LLMs through Semiotics</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      This paper challenges the prevailing tendency to frame Large Language Models (LLMs) as cognitive systems, arguing instead for a semiotic perspective that situates these models within the broader dynamics of sign manipulation and meaning-making. Rather than assuming that LLMs understand language or simulate human thought, we propose that their primary function is to recombine, recontextualize, and circulate linguistic forms based on probabilistic associations. By shifting from a cognitivist to a semiotic framework, we avoid anthropomorphism and gain a more precise understanding of how LLMs participate in cultural processes, not by thinking, but by generating texts that invite interpretation. Through theoretical analysis and practical examples, the paper demonstrates how LLMs function as semiotic agents whose outputs can be treated as interpretive acts, open to contextual negotiation and critical reflection. We explore applications in literature, philosophy, education, and cultural production, emphasizing how LLMs can serve as tools for creativity, dialogue, and critical inquiry. The semiotic paradigm foregrounds the situated, contingent, and socially embedded nature of meaning, offering a more rigorous and ethically aware framework for studying and using LLMs. Ultimately, this approach reframes LLMs as technological participants in an ongoing ecology of signs. They do not possess minds, but they alter how we read, write, and make meaning, compelling us to reconsider the foundations of language, interpretation, and the role of artificial systems in the production of knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19676v2">A Survey of LLM-Driven AI Agent Communication: Protocols, Security Risks, and Defense Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      In recent years, Large-Language-Model-driven AI agents have exhibited unprecedented intelligence and adaptability, and are rapidly changing human production and life. Nowadays, agents are undergoing a new round of evolution. They no longer act as an isolated island like LLMs. Instead, they start to communicate with diverse external entities, such as other agents and tools, to perform more complex tasks collectively. Under this trend, agent communication is regarded as a foundational pillar of the future AI ecosystem, and many organizations have intensively begun to design related communication protocols (e.g., Anthropic's MCP and Google's A2A) within the recent few months. However, this new field exposes significant security hazards, which can cause severe damage to real-world scenarios. To help researchers quickly figure out this promising topic and benefit the future agent communication development, this paper presents a comprehensive survey of agent communication security. More precisely, we first present a clear definition of agent communication and categorize the entire lifecycle of agent communication into three stages: user-agent interaction, agent-agent communication, and agent-environment communication. Next, for each communication phase, we dissect related protocols and analyze the security risks according to the communication characteristics. Then, we summarize and outlook on the possible defense countermeasures for each risk. In addition, we conduct experiments using MCP and A2A to help readers better understand the novel vulnerabilities brought by agent communication. Finally, we discuss open issues and future directions in this promising research field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06432v2">Integrating Expert Labels into LLM-based Emission Goal Detection: Example Selection vs Automatic Prompt Design</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      We address the detection of emission reduction goals in corporate reports, an important task for monitoring companies' progress in addressing climate change. Specifically, we focus on the issue of integrating expert feedback in the form of labeled example passages into LLM-based pipelines, and compare the two strategies of (1) a dynamic selection of few-shot examples and (2) the automatic optimization of the prompt by the LLM itself. Our findings on a public dataset of 769 climate-related passages from real-world business reports indicate that automatic prompt optimization is the superior approach, while combining both methods provides only limited benefit. Qualitative results indicate that optimized prompts do indeed capture many intricacies of the targeted emission goal extraction task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23520v2">ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      With the increasing interest in robotic synthesis in the context of organic chemistry, the automated extraction of chemical procedures from literature is critical. However, this task remains challenging due to the inherent ambiguity of chemical language and the high cost of human annotation required for developing reliable computer-aided extraction protocols. Here, we present ChemActor, a fully fine-tuned large language model (LLM), as a chemical executor to convert between unstructured experimental procedures and structured action sequences. We propose a sequential LLM-generated data framework to address the challenges of insufficient and low-quality annotated data. This framework integrates a data selection module that selects data based on distribution divergence, with a general-purpose LLM, to generate machine-executable actions from a single molecule input. Additionally, we introduce a novel multi-round LLMs circle review metric, which reflects the model's advanced understanding of chemical experimental procedures. Extensive experiments on reaction-to-description (R2D) and description-to-action (D2A) tasks demonstrate that ChemActor, augmented by LLM-generated data, achieves state-of-the-art performance, outperforming the baseline model by 10%. The code is available at: https://github.com/Zhanghahah/ChemActor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21393v3">An evaluation of LLMs and Google Translate for translation of selected Indian languages via sentiment and semantic analyses</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have been prominent for language translation, including low-resource languages. There has been limited study on the assessment of the quality of translations generated by LLMs, including Gemini, GPT, and Google Translate. This study addresses this limitation by using semantic and sentiment analysis of selected LLMs for Indian languages, including Sanskrit, Telugu and Hindi. We select prominent texts (Bhagavad Gita, Tamas and Maha Prasthanam ) that have been well translated by experts and use LLMs to generate their translations into English, and provide a comparison with selected expert (human) translations. Our investigation revealed that while LLMs have made significant progress in translation accuracy, challenges remain in preserving sentiment and semantic integrity, especially in metaphorical and philosophical contexts for texts such as the Bhagavad Gita. The sentiment analysis revealed that GPT models are better at preserving the sentiment polarity for the given texts when compared to human (expert) translation. The results revealed that GPT models are generally better at maintaining the sentiment and semantics when compared to Google Translate. This study could help in the development of accurate and culturally sensitive translation systems for large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21248v2">ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated potential in assisting scientific research, yet their ability to discover high-quality research hypotheses remains unexamined due to the lack of a dedicated benchmark. To address this gap, we introduce the first large-scale benchmark for evaluating LLMs with a near-sufficient set of sub-tasks of scientific discovery: inspiration retrieval, hypothesis composition, and hypothesis ranking. We develop an automated framework that extracts critical components - research questions, background surveys, inspirations, and hypotheses - from scientific papers across 12 disciplines, with expert validation confirming its accuracy. To prevent data contamination, we focus exclusively on papers published in 2024, ensuring minimal overlap with LLM pretraining data. Our evaluation reveals that LLMs perform well in retrieving inspirations, an out-of-distribution task, suggesting their ability to surface novel knowledge associations. This positions LLMs as "research hypothesis mines", capable of facilitating automated scientific discovery by generating innovative hypotheses at scale with minimal human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18232v2">Two-Stage Regularization-Based Structured Pruning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) is largely hindered by their large number of parameters. Structural pruning has emerged as a promising solution. Prior structured pruning methods directly remove unimportant parameters based on certain metrics, which often causes knowledge loss and necessitates extensive retraining. To overcome this, we introduce a novel pruning method TRSP: Two-Stage Regularization-Based Structured Pruning for LLMs. Specifically, we multiply the output of each transformer layer by an initial learnable weight and iteratively learn these weights by adding their $\ell_1$-norm as a regularization term to the loss function, serving as the first-stage regularization. Subsequently, we apply additional regularization to the difference between the output and input of layers with smaller weights, encouraging the shift of knowledge to the preserved layers. This serves as the second-stage regularization. TRSP retains more knowledge and better preserves model performance than direct parameter elimination. Through extensive experimentation we show that TRSP outperforms strong layer-wise structured pruning methods without requiring retraining. As a layer-wise pruning method, it delivers notable end-to-end acceleration, making it a promising solution for efficient LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01144v4">BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      The rapidly increasing size of large language models (LLMs) presents significant challenges in memory usage and computational costs. Quantizing both weights and activations can address these issues, with hardware-supported fine-grained scaling emerging as a promising solution to mitigate outliers. However, existing methods struggle to capture nuanced block data distributions. We propose BlockDialect, a block-wise fine-grained mixed format technique that assigns a per-block optimal number format from a formatbook for better data representation. Additionally, we introduce DialectFP4, a formatbook of FP4 variants (akin to dialects) that adapt to diverse data distributions. To leverage this efficiently, we propose a two-stage approach for online DialectFP4 activation quantization. Importantly, DialectFP4 ensures energy efficiency by selecting representable values as scaled integers compatible with low-precision integer arithmetic. BlockDialect achieves 10.78% (7.48%) accuracy gain on the LLaMA3-8B (LLaMA2-7B) model compared to MXFP4 format with lower bit usage per data, while being only 5.45% (2.69%) below full precision even when quantizing full-path matrix multiplication. Focusing on how to represent over how to scale, our work presents a promising path for energy-efficient LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.02277v3">Junk DNA Hypothesis: Pruning Small Pre-Trained Weights Irreversibly and Monotonically Impairs "Difficult" Downstream Tasks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Published at ICML 2024
    </div>
    <details class="paper-abstract">
      We present Junk DNA Hypothesis by adopting a novel task-centric angle for the pre-trained weights of large language models (LLMs). It has been believed that weights in LLMs contain significant redundancy, leading to the conception that a considerable chunk of the parameters can be removed by pruning without compromising performance. Contrary to this belief, this paper presents a counter-argument: small-magnitude weights of pre-trained model weights encode vital knowledge essential for tackling difficult downstream tasks - manifested as the monotonic relationship between the performance drop of downstream tasks across the difficulty spectrum, as we prune more pre-trained weights by magnitude. Moreover, we reveal that these seemingly inconsequential weights can result in irreparable loss of knowledge and performance degradation in difficult tasks, even when downstream continual training is allowed. Interestingly, our evaluations show that the other popular compression, namely quantization, fails to exhibit similar monotonic effect and does not as convincingly disentangle this task-difficulty information. To study formally, we introduce several quantifiable metrics to gauge the downstream task difficulty: (1) within the same task category, and (2) across different task categories. Our extensive experiments substantiate the Junk DNA Hypothesis across a diverse range of model sizes, tasks, datasets, and even pruning methods. Codes are available at: https://github.com/VITA-Group/Junk_DNA_Hypothesis.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01241v1">Beyond First-Order: Training LLMs with Stochastic Conjugate Subgradients and AdamW</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Stochastic gradient-based descent (SGD), have long been central to training large language models (LLMs). However, their effectiveness is increasingly being questioned, particularly in large-scale applications where empirical evidence suggests potential performance limitations. In response, this paper proposes a stochastic conjugate subgradient method together with adaptive sampling tailored specifically for training LLMs. The method not only achieves faster convergence per iteration but also demonstrates improved scalability compared to traditional SGD techniques. It leverages sample complexity analysis to adaptively choose the sample size, employs a stochastic conjugate subgradient approach to determine search directions and utilizing an AdamW-like algorithm to adaptively adjust step sizes. This approach preserves the key advantages of first-order methods while effectively addressing the nonconvexity and non-smoothness inherent in LLMs training. Additionally, we provide a detailed analysis of the advantage of the algorithm. Experimental results show that the proposed method not only maintains, but in many cases surpasses, the scalability of traditional SGD techniques, significantly enhancing both the speed and accuracy of the optimization process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13757v4">GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01216v1">PAE MobiLLM: Privacy-Aware and Efficient LLM Fine-Tuning on the Mobile Device via Additive Side-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      There is a huge gap between numerous intriguing applications fostered by on-device large language model (LLM) fine-tuning (FT) from fresh mobile data and the limited resources of a mobile device. While existing server-assisted methods (e.g., split learning or side-tuning) may enable LLM FT on the local mobile device, they suffer from heavy communication burdens of activation transmissions, and may disclose data, labels or fine-tuned models to the server. To address those issues, we develop PAE MobiLLM, a privacy-aware and efficient LLM FT method which can be deployed on the mobile device via server-assisted additive side-tuning. To further accelerate FT convergence and improve computing efficiency, PAE MobiLLM integrates activation caching on the server side, which allows the server to reuse historical activations and saves the mobile device from repeatedly computing forward passes for the recurring data samples. Besides, to reduce communication cost, PAE MobiLLM develops a one-token (i.e., ``pivot'' token) activation shortcut that transmits only a single activation dimension instead of full activation matrices to guide the side network tuning. Last but not least, PAE MobiLLM introduces the additive adapter side-network design which makes the server train the adapter modules based on device-defined prediction differences rather than raw ground-truth labels. In this way, the server can only assist device-defined side-network computing, and learn nothing about data, labels or fine-tuned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01206v1">2024 NASA SUITS Report: LLM-Driven Immersive Augmented Reality User Interface for Robotics and Space Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      As modern computing advances, new interaction paradigms have emerged, particularly in Augmented Reality (AR), which overlays virtual interfaces onto physical objects. This evolution poses challenges in machine perception, especially for tasks like 3D object pose estimation in complex, dynamic environments. Our project addresses critical issues in human-robot interaction within mobile AR, focusing on non-intrusive, spatially aware interfaces. We present URSA, an LLM-driven immersive AR system developed for NASA's 2023-2024 SUITS challenge, targeting future spaceflight needs such as the Artemis missions. URSA integrates three core technologies: a head-mounted AR device (e.g., HoloLens) for intuitive visual feedback, voice control powered by large language models for hands-free interaction, and robot tracking algorithms that enable accurate 3D localization in dynamic settings. To enhance precision, we leverage digital twin localization technologies, using datasets like DTTD-Mobile and specialized hardware such as the ZED2 camera for real-world tracking under noise and occlusion. Our system enables real-time robot control and monitoring via an AR interface, even in the absence of ground-truth sensors--vital for hazardous or remote operations. Key contributions include: (1) a non-intrusive AR interface with LLM-based voice input; (2) a ZED2-based dataset tailored for non-rigid robotic bodies; (3) a Local Mission Control Console (LMCC) for mission visualization; (4) a transformer-based 6DoF pose estimator (DTTDNet) optimized for depth fusion and real-time tracking; and (5) end-to-end integration for astronaut mission support. This work advances digital twin applications in robotics, offering scalable solutions for both aerospace and industrial domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20295v2">Self-reflective Uncertainties: Do LLMs Know Their Internal Answer Distribution?</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      To reveal when a large language model (LLM) is uncertain about a response, uncertainty quantification commonly produces percentage numbers along with the output. But is this all we can do? We argue that in the output space of LLMs, the space of strings, exist strings expressive enough to summarize the distribution over output strings the LLM deems possible. We lay a foundation for this new avenue of uncertainty explication and present SelfReflect, a theoretically-motivated metric to assess how faithfully a string summarizes an LLM's internal answer distribution. We show that SelfReflect is able to discriminate even subtle differences of candidate summary strings and that it aligns with human judgement, outperforming alternative metrics such as LLM judges and embedding comparisons. With SelfReflect, we investigate a number of self-summarization methods and find that even state-of-the-art reasoning models struggle to explicate their internal uncertainty. But we find that faithful summarizations can be generated by sampling and summarizing. To support the development of this universal form of LLM uncertainties, we publish our metric at https://github.com/apple/ml-selfreflect
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00979v1">Enhancing LLM Agent Safety via Causal Influence Prompting</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Accepted at ACL 2025 Findings, Source code: https://github.com/HahmDY/causal_influence_prompting.git
    </div>
    <details class="paper-abstract">
      As autonomous agents powered by large language models (LLMs) continue to demonstrate potential across various assistive tasks, ensuring their safe and reliable behavior is crucial for preventing unintended consequences. In this work, we introduce CIP, a novel technique that leverages causal influence diagrams (CIDs) to identify and mitigate risks arising from agent decision-making. CIDs provide a structured representation of cause-and-effect relationships, enabling agents to anticipate harmful outcomes and make safer decisions. Our approach consists of three key steps: (1) initializing a CID based on task specifications to outline the decision-making process, (2) guiding agent interactions with the environment using the CID, and (3) iteratively refining the CID based on observed behaviors and outcomes. Experimental results demonstrate that our method effectively enhances safety in both code execution and mobile device control tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01080v1">Development and Comparative Evaluation of Three Artificial Intelligence Models (NLP, LLM, JEPA) for Predicting Triage in Emergency Departments: A 7-Month Retrospective Proof-of-Concept</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Triage errors, including undertriage and overtriage, are persistent challenges in emergency departments (EDs). With increasing patient influx and staff shortages, the integration of artificial intelligence (AI) into triage protocols has gained attention. This study compares the performance of three AI models [Natural Language Processing (NLP), Large Language Models (LLM), and Joint Embedding Predictive Architecture (JEPA)] in predicting triage outcomes against the FRENCH scale and clinical practice.We conducted a retrospective analysis of a prospectively recruited cohort gathering adult patient triage data over a 7-month period at the Roger Salengro Hospital ED (Lille, France). Three AI models were trained and validated : (1) TRIAGEMASTER (NLP), (2) URGENTIAPARSE (LLM), and (3) EMERGINET (JEPA). Data included demographic details, verbatim chief complaints, vital signs, and triage outcomes based on the FRENCH scale and GEMSA coding. The primary outcome was the concordance of AI-predicted triage level with the FRENCH gold-standard. It was assessed thanks to various indicators : F1-Score, Weighted Kappa, Spearman, MAE, RMSE. The LLM model (URGENTIAPARSE) showed higher accuracy (composite score: 2.514) compared to JEPA (EMERGINET, 0.438) and NLP (TRIAGEMASTER, -3.511), outperforming nurse triage (-4.343). Secondary analyses highlighted the effectiveness of URGENTIAPARSE in predicting hospitalization needs (GEMSA) and its robustness with structured data versus raw transcripts (either for GEMSA prediction or for FRENCH prediction). LLM architecture, through abstraction of patient representations, offers the most accurate triage predictions among tested models. Integrating AI into ED workflows could enhance patient safety and operational efficiency, though integration into clinical workflows requires addressing model limitations and ensuring ethical transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00838v1">Stylometry recognizes human and LLM-generated texts in short samples</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      The paper explores stylometry as a method to distinguish between texts created by Large Language Models (LLMs) and humans, addressing issues of model attribution, intellectual property, and ethical AI use. Stylometry has been used extensively to characterise the style and attribute authorship of texts. By applying it to LLM-generated texts, we identify their emergent writing patterns. The paper involves creating a benchmark dataset based on Wikipedia, with (a) human-written term summaries, (b) texts generated purely by LLMs (GPT-3.5/4, LLaMa 2/3, Orca, and Falcon), (c) processed through multiple text summarisation methods (T5, BART, Gensim, and Sumy), and (d) rephrasing methods (Dipper, T5). The 10-sentence long texts were classified by tree-based models (decision trees and LightGBM) using human-designed (StyloMetrix) and n-gram-based (our own pipeline) stylometric features that encode lexical, grammatical, syntactic, and punctuation patterns. The cross-validated results reached a performance of up to .87 Matthews correlation coefficient in the multiclass scenario with 7 classes, and accuracy between .79 and 1. in binary classification, with the particular example of Wikipedia and GPT-4 reaching up to .98 accuracy on a balanced dataset. Shapley Additive Explanations pinpointed features characteristic of the encyclopaedic text type, individual overused words, as well as a greater grammatical standardisation of LLMs with respect to human-written texts. These results show -- crucially, in the context of the increasingly sophisticated LLMs -- that it is possible to distinguish machine- from human-generated texts at least for a well-defined text type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00833v1">HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Project Page: https://openhumanoidgen.github.io
    </div>
    <details class="paper-abstract">
      For robotic manipulation, existing robotics datasets and simulation benchmarks predominantly cater to robot-arm platforms. However, for humanoid robots equipped with dual arms and dexterous hands, simulation tasks and high-quality demonstrations are notably lacking. Bimanual dexterous manipulation is inherently more complex, as it requires coordinated arm movements and hand operations, making autonomous data collection challenging. This paper presents HumanoidGen, an automated task creation and demonstration collection framework that leverages atomic dexterous operations and LLM reasoning to generate relational constraints. Specifically, we provide spatial annotations for both assets and dexterous hands based on the atomic operations, and perform an LLM planner to generate a chain of actionable spatial constraints for arm movements based on object affordances and scenes. To further improve planning ability, we employ a variant of Monte Carlo tree search to enhance LLM reasoning for long-horizon tasks and insufficient annotation. In experiments, we create a novel benchmark with augmented scenarios to evaluate the quality of the collected data. The results show that the performance of the 2D and 3D diffusion policies can scale with the generated dataset. Project page is https://openhumanoidgen.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00829v1">On the Surprising Efficacy of LLMs for Penetration-Testing</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      This paper presents a critical examination of the surprising efficacy of Large Language Models (LLMs) in penetration testing. The paper thoroughly reviews the evolution of LLMs and their rapidly expanding capabilities which render them increasingly suitable for complex penetration testing operations. It systematically details the historical adoption of LLMs in both academic research and industry, showcasing their application across various offensive security tasks and covering broader phases of the cyber kill chain. Crucially, the analysis also extends to the observed adoption of LLMs by malicious actors, underscoring the inherent dual-use challenge of this technology within the security landscape. The unexpected effectiveness of LLMs in this context is elucidated by several key factors: the strong alignment between penetration testing's reliance on pattern-matching and LLMs' core strengths, their inherent capacity to manage uncertainty in dynamic environments, and cost-effective access to competent pre-trained models through LLM providers. The current landscape of LLM-aided penetration testing is categorized into interactive 'vibe-hacking' and the emergence of fully autonomous systems. The paper identifies and discusses significant obstacles impeding wider adoption and safe deployment. These include critical issues concerning model reliability and stability, paramount safety and security concerns, substantial monetary and ecological costs, implications for privacy and digital sovereignty, complex questions of accountability, and profound ethical dilemmas. This comprehensive review and analysis provides a foundation for discussion on future research directions and the development of robust safeguards at the intersection of AI and security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01077v1">Good Enough to Learn: LLM-based Anomaly Detection in ECU Logs without Reliable Labels</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 6 pages, 7 figures, 4 tables, accepted to IEEE Intelligent Vehicles Symposium (IV) 2025
    </div>
    <details class="paper-abstract">
      Anomaly detection often relies on supervised or clustering approaches, with limited success in specialized domains like automotive communication systems where scalable solutions are essential. We propose a novel decoder-only Large Language Model (LLM) to detect anomalies in Electronic Control Unit (ECU) communication logs. Our approach addresses two key challenges: the lack of LLMs tailored for ECU communication and the complexity of inconsistent ground truth data. By learning from UDP communication logs, we formulate anomaly detection simply as identifying deviations in time from normal behavior. We introduce an entropy regularization technique that increases model's uncertainty in known anomalies while maintaining consistency in similar scenarios. Our solution offers three novelties: a decoder-only anomaly detection architecture, a way to handle inconsistent labeling, and an adaptable LLM for different ECU communication use cases. By leveraging the generative capabilities of decoder-only models, we present a new technique that addresses the high cost and error-prone nature of manual labeling through a more scalable system that is able to learn from a minimal set of examples, while improving detection accuracy in complex communication environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00814v1">Many LLMs Are More Utilitarian Than One</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 9 pages, 8 Figures, 7 tables
    </div>
    <details class="paper-abstract">
      Moral judgment is integral to large language model (LLM) alignment and social reasoning. As multi-agent systems gain prominence, it becomes crucial to understand how LLMs function collectively during collaboration, compared to individual agents. In human moral judgment, group deliberation leads to a utilitarian boost: a tendency to endorse norm violations that maximize benefits for the greatest number of people despite harms. We study whether a similar dynamic emerges in multi-agent LLM systems. We tested six models on well-established sets of moral dilemmas across two conditions: (1) Solo, where models reasoned independently, and (2) Group, where they engaged in multi-turn discussions in pairs or triads. In personal moral dilemmas, where agents must decide to directly harm one individual to maximize the utility for others, all models found moral violations to be more acceptable when part of a group than individually, similar to human experiments. Some models endorsed actions that maximized overall well-being, even if they benefited strangers over familiar individuals. Others became more willing to violate moral norms in groups. However, while human groups show a similar action bias, the mechanism for their utilitarian boost differs from LLMs. Whereas the human shift comes from heightened sensitivity to decision outcomes, LLM groups show either reduced norm sensitivity or enhanced impartiality. This suggests that while the surface behavior of LLM collectives mimics human group reasoning, the underlying drivers differ. We discuss the implications for AI alignment, multi-agent design, and artificial moral reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00797v1">VEDA: Efficient LLM Generation Through Voting-based KV Cache Eviction and Dataflow-flexible Accelerator</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 DAC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in natural language processing tasks but pose significant computational and memory challenges for edge deployment due to their intensive resource demands. This work addresses the efficiency of LLM inference by algorithm-hardware-dataflow tri-optimizations. We propose a novel voting-based KV cache eviction algorithm, balancing hardware efficiency and algorithm accuracy by adaptively identifying unimportant kv vectors. From a dataflow perspective, we introduce a flexible-product dataflow and a runtime reconfigurable PE array for matrix-vector multiplication. The proposed approach effectively handles the diverse dimensional requirements and solves the challenges of incrementally varying sequence lengths. Additionally, an element-serial scheduling scheme is proposed for nonlinear operations, such as softmax and layer normalization (layernorm). Results demonstrate a substantial reduction in latency, accompanied by a significant decrease in hardware complexity, from O(N) to O(1). The proposed solution is realized in a custom-designed accelerator, VEDA, which outperforms existing hardware platforms. This research represents a significant advancement in LLM inference on resource-constrained edge devices, facilitating real-time processing, enhancing data privacy, and enabling model customization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00754v1">Language-Unlocked ViT (LUViT): Empowering Self-Supervised Vision Transformers with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 26 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The integration of Large Language Model (LLMs) blocks with Vision Transformers (ViTs) holds immense promise for vision-only tasks by leveraging the rich semantic knowledge and reasoning capabilities of LLMs. However, a fundamental challenge lies in the inherent modality mismatch between text-centric pretraining of LLMs and vision-centric training of ViTs. Direct fusion often fails to fully exploit the LLM's potential and suffers from unstable finetuning. As a result, LLM blocks are kept frozen while only the vision components are learned. As a remedy to these challenges, we introduce Language-Unlocked Vision Transformers (LUViT), a novel approach that bridges this modality mismatch through a synergistic pre-training strategy. LUViT co-adapts a ViT backbone and an LLM fusion block by (1) employing Masked Auto-Encoding (MAE) to pre-train the ViT for richer visual representations, and (2) concurrently training Low-Rank Adaptation (LoRA) layers within the LLM block using the MAE objective. This joint optimization guides the ViT to produce LLM-aligned features and the LLM to effectively interpret visual information. We demonstrate through extensive experiments that LUViT significantly improves performance on various downstream vision tasks, showcasing a more effective and efficient pathway to harness LLM knowledge for visual understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00742v1">Evaluating LLMs and Prompting Strategies for Automated Hardware Diagnosis from Textual User-Reports</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 To be published in the Proceedings of the Brazilian Integrated Software and Hardware Seminar 2025 (SEMISH 2025)
    </div>
    <details class="paper-abstract">
      Computer manufacturers offer platforms for users to describe device faults using textual reports such as "My screen is flickering". Identifying the faulty component from the report is essential for automating tests and improving user experience. However, such reports are often ambiguous and lack detail, making this task challenging. Large Language Models (LLMs) have shown promise in addressing such issues. This study evaluates 27 open-source models (1B-72B parameters) and 2 proprietary LLMs using four prompting strategies: Zero-Shot, Few-Shot, Chain-of-Thought (CoT), and CoT+Few-Shot (CoT+FS). We conducted 98,948 inferences, processing over 51 million input tokens and generating 13 million output tokens. We achieve f1-score up to 0.76. Results show that three models offer the best balance between size and performance: mistral-small-24b-instruct and two smaller models, llama-3.2-1b-instruct and gemma-2-2b-it, that offer competitive performance with lower VRAM usage, enabling efficient inference on end-user devices as modern laptops or smartphones with NPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00715v1">EARN: Efficient Inference Acceleration for LLM-based Generative Recommendation by Register Tokens</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Accepted by KDD 2025
    </div>
    <details class="paper-abstract">
      Large Language Model-based generative recommendation (LLMRec) has achieved notable success, but it suffers from high inference latency due to massive computational overhead and memory pressure of KV Cache. Existing KV Cache reduction methods face critical limitations: cache compression offers marginal acceleration given recommendation tasks' short decoding steps, while prompt compression risks discarding vital interaction history. Through systematic analysis of attention patterns in LLMRec, we uncover two pivotal insights: 1) layer-wise attention sparsity inversion where early layers retain dense informative patterns while later layers exhibit high redundancy, and 2) dual attention sinks phenomenon where attention scores concentrate on both head and tail tokens of input sequences. Motivated by these insights, we propose EARN, an efficient inference framework that leverages the early layers to compress information into register tokens placed at the input sequence boundaries, then focuses solely on these tokens in the subsequent layers. Extensive experiments on three datasets, two LLMRec methods and two LLM architectures demonstrate EARN's superiority, achieving up to 3.79x speedup and 80.8% KV Cache reduction with better accuracy than the general finetuning approach. Our work bridges the efficiency-effectiveness gap in LLMRec, offering practical deployment advantages for industrial scenarios.
    </details>
</div>
