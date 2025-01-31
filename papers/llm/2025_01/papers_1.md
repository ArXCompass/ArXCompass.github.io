# llm - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18596v1">DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      We introduce DeltaLLM, a new post-training compression technique to reduce the memory footprint of LLMs. We propose an alternative way of structuring LLMs with weight sharing between layers in subsequent Transformer blocks, along with additional low-rank difference matrices between them. For training, we adopt the progressing module replacement method and show that the lightweight training of the low-rank modules with approximately 30M-40M tokens is sufficient to achieve performance on par with LLMs of comparable sizes trained from scratch. We release the resultant models, DeltaLLAMA and DeltaPHI, with a 12% parameter reduction, retaining 90% of the performance of the base Llama and Phi models on common knowledge and reasoning benchmarks. Our method also outperforms compression techniques JointDrop, LaCo, ShortGPT and SliceGPT with the same number of parameters removed. For example, DeltaPhi 2.9B with a 24% reduction achieves similar average zero-shot accuracies as recovery fine-tuned SlicedPhi 3.3B with a 12% reduction, despite being approximately 400M parameters smaller with no fine-tuning applied. This work provides new insights into LLM architecture design and compression methods when storage space is critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18585v1">Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as OpenAI's o1 have demonstrated remarkable abilities in complex reasoning tasks by scaling test-time compute and exhibiting human-like deep thinking. However, we identify a phenomenon we term underthinking, where o1-like LLMs frequently switch between different reasoning thoughts without sufficiently exploring promising paths to reach a correct solution. This behavior leads to inadequate depth of reasoning and decreased performance, particularly on challenging mathematical problems. To systematically analyze this issue, we conduct experiments on three challenging test sets and two representative open-source o1-like models, revealing that frequent thought switching correlates with incorrect responses. We introduce a novel metric to quantify underthinking by measuring token efficiency in incorrect answers. To address underthinking, we propose a decoding strategy with thought switching penalty TIP that discourages premature transitions between thoughts, encouraging deeper exploration of each reasoning path. Experimental results demonstrate that our approach improves accuracy across challenging datasets without requiring model fine-tuning. Our findings contribute to understanding reasoning inefficiencies in o1-like LLMs and offer a practical solution to enhance their problem-solving capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18539v1">Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      Real-world open-domain questions can be complicated, particularly when answering them involves information from multiple information sources. LLMs have demonstrated impressive performance in decomposing complex tasks into simpler steps, and previous work has used it for better retrieval in support of complex questions. However, LLM's decomposition of questions is unaware of what data is available and how data is organized, often leading to a sub-optimal retrieval performance. Recent effort in agentic RAG proposes to perform retrieval in an iterative fashion, where a followup query is derived as an action based on previous rounds of retrieval. While this provides one way of interacting with the data collection, agentic RAG's exploration of data is inefficient because successive queries depend on previous results rather than being guided by the organization of available data in the collection. To address this problem, we propose an LLM-based retrieval method -- ARM, that aims to better align the question with the organization of the data collection by exploring relationships among data objects beyond matching the utterance of the query, thus leading to a retrieve-all-at-once solution for complex queries. We evaluated ARM on two datasets, Bird and OTT-QA. On Bird, it outperforms standard RAG with query decomposition by up to 5.2 pt in execution accuracy and agentic RAG (ReAct) by up to 15.9 pt. On OTT-QA, it achieves up to 5.5 pt and 19.3 pt higher F1 match scores compared to these approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18536v1">Illusions of Relevance: Using Content Injection Attacks to Deceive Retrievers, Rerankers, and LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      Consider a scenario in which a user searches for information, only to encounter texts flooded with misleading or non-relevant content. This scenario exemplifies a simple yet potent vulnerability in neural Information Retrieval (IR) pipelines: content injection attacks. We find that embedding models for retrieval, rerankers, and large language model (LLM) relevance judges are vulnerable to these attacks, in which adversaries insert misleading text into passages to manipulate model judgements. We identify two primary threats: (1) inserting unrelated or harmful content within passages that still appear deceptively "relevant", and (2) inserting entire queries or key query terms into passages to boost their perceived relevance. While the second tactic has been explored in prior research, we present, to our knowledge, the first empirical analysis of the first threat, demonstrating how state-of-the-art models can be easily misled. Our study systematically examines the factors that influence an attack's success, such as the placement of injected content and the balance between relevant and non-relevant material. Additionally, we explore various defense strategies, including adversarial passage classifiers, retriever fine-tuning to discount manipulated content, and prompting LLM judges to adopt a more cautious approach. However, we find that these countermeasures often involve trade-offs, sacrificing effectiveness for attack robustness and sometimes penalizing legitimate documents in the process. Our findings highlight the need for stronger defenses against these evolving adversarial strategies to maintain the trustworthiness of IR systems. We release our code and scripts to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17141v3">Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 Main Paper 1-9 pages, Supplementary Materials: 10-17, 13 figures
    </div>
    <details class="paper-abstract">
      Hacking poses a significant threat to cybersecurity, inflicting billions of dollars in damages annually. To mitigate these risks, ethical hacking, or penetration testing, is employed to identify vulnerabilities in systems and networks. Recent advancements in large language models (LLMs) have shown potential across various domains, including cybersecurity. However, there is currently no comprehensive, open, end-to-end automated penetration testing benchmark to drive progress and evaluate the capabilities of these models in security contexts. This paper introduces a novel open benchmark for LLM-based automated penetration testing, addressing this critical gap. We first evaluate the performance of LLMs, including GPT-4o and Llama 3.1-405B, using the state-of-the-art PentestGPT tool. Our findings reveal that while Llama 3.1 demonstrates an edge over GPT-4o, both models currently fall short of performing fully automated, end-to-end penetration testing. Next, we advance the state-of-the-art and present ablation studies that provide insights into improving the PentestGPT tool. Our research illuminates the challenges LLMs face in each aspect of Pentesting, e.g. enumeration, exploitation, and privilege escalation. This work contributes to the growing body of knowledge on AI-assisted cybersecurity and lays the foundation for future research in automated penetration testing using large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18492v1">GuardReasoner: Towards Reasoning-based LLM Safeguards</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 22 pages, 18 figures
    </div>
    <details class="paper-abstract">
      As LLMs increasingly impact safety-critical applications, ensuring their safety using guardrails remains a key challenge. This paper proposes GuardReasoner, a new safeguard for LLMs, by guiding the guard model to learn to reason. Concretely, we first create the GuardReasonerTrain dataset, which consists of 127K samples with 460K detailed reasoning steps. Then, we introduce reasoning SFT to unlock the reasoning capability of guard models. In addition, we present hard sample DPO to further strengthen their reasoning ability. In this manner, GuardReasoner achieves better performance, explainability, and generalizability. Extensive experiments and analyses on 13 benchmarks of 3 guardrail tasks demonstrate its superiority. Remarkably, GuardReasoner 8B surpasses GPT-4o+CoT by 5.74% and LLaMA Guard 3 8B by 20.84% F1 score on average. We release the training data, code, and models with different scales (1B, 3B, 8B) of GuardReasoner : https://github.com/yueliu1999/GuardReasoner/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07291v2">Evaluating LLM-based Personal Information Extraction and Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 To appear in USENIX Security Symposium 2025
    </div>
    <details class="paper-abstract">
      Automatically extracting personal information--such as name, phone number, and email address--from publicly available profiles at a large scale is a stepstone to many other security attacks including spear phishing. Traditional methods--such as regular expression, keyword search, and entity detection--achieve limited success at such personal information extraction. In this work, we perform a systematic measurement study to benchmark large language model (LLM) based personal information extraction and countermeasures. Towards this goal, we present a framework for LLM-based extraction attacks; collect four datasets including a synthetic dataset generated by GPT-4 and three real-world datasets with manually labeled eight categories of personal information; introduce a novel mitigation strategy based on prompt injection; and systematically benchmark LLM-based attacks and countermeasures using ten LLMs and five datasets. Our key findings include: LLM can be misused by attackers to accurately extract various personal information from personal profiles; LLM outperforms traditional methods; and prompt injection can defend against strong LLM-based attacks, reducing the attack to less effective traditional ones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18475v1">CLoQ: Enhancing Fine-Tuning of Quantized LLMs via Calibrated LoRA Initialization</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using low-rank adaptation (LoRA) has become a highly efficient approach for downstream tasks, particularly in scenarios with limited computational resources. However, applying LoRA techniques to quantized LLMs poses unique challenges due to the reduced representational precision of quantized weights. In this paper, we introduce CLoQ (Calibrated LoRA initialization for Quantized LLMs), a simplistic initialization strategy designed to overcome these challenges. Our approach focuses on minimizing the layer-wise discrepancy between the original LLM and its quantized counterpart with LoRA components during initialization. By leveraging a small calibration dataset, CLoQ quantizes a pre-trained LLM and determines the optimal LoRA components for each layer, ensuring a strong foundation for subsequent fine-tuning. A key contribution of this work is a novel theoretical result that enables the accurate and closed-form construction of these optimal LoRA components. We validate the efficacy of CLoQ across multiple tasks such as language generation, arithmetic reasoning, and commonsense reasoning, demonstrating that it consistently outperforms existing LoRA fine-tuning methods for quantized LLMs, especially at ultra low-bit widths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16673v2">LLM-AutoDiff: Auto-Differentiate Any LLM Workflow</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have reshaped natural language processing, powering applications from multi-hop retrieval and question answering to autonomous agent workflows. Yet, prompt engineering -- the task of crafting textual inputs to effectively direct LLMs -- remains difficult and labor-intensive, particularly for complex pipelines that combine multiple LLM calls with functional operations like retrieval and data formatting. We introduce LLM-AutoDiff: a novel framework for Automatic Prompt Engineering (APE) that extends textual gradient-based methods (such as Text-Grad) to multi-component, potentially cyclic LLM architectures. Implemented within the AdalFlow library, LLM-AutoDiff treats each textual input as a trainable parameter and uses a frozen backward engine LLM to generate feedback-akin to textual gradients -- that guide iterative prompt updates. Unlike prior single-node approaches, LLM-AutoDiff inherently accommodates functional nodes, preserves time-sequential behavior in repeated calls (e.g., multi-hop loops), and combats the "lost-in-the-middle" problem by isolating distinct sub-prompts (instructions, formats, or few-shot examples). It further boosts training efficiency by focusing on error-prone samples through selective gradient computation. Across diverse tasks, including single-step classification, multi-hop retrieval-based QA, and agent-driven pipelines, LLM-AutoDiff consistently outperforms existing textual gradient baselines in both accuracy and training cost. By unifying prompt optimization through a graph-centric lens, LLM-AutoDiff offers a powerful new paradigm for scaling and automating LLM workflows - mirroring the transformative role that automatic differentiation libraries have long played in neural network research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16727v2">xJailbreak: Representation Space Guided Reinforcement Learning for Interpretable LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      Safety alignment mechanism are essential for preventing large language models (LLMs) from generating harmful information or unethical content. However, cleverly crafted prompts can bypass these safety measures without accessing the model's internal parameters, a phenomenon known as black-box jailbreak. Existing heuristic black-box attack methods, such as genetic algorithms, suffer from limited effectiveness due to their inherent randomness, while recent reinforcement learning (RL) based methods often lack robust and informative reward signals. To address these challenges, we propose a novel black-box jailbreak method leveraging RL, which optimizes prompt generation by analyzing the embedding proximity between benign and malicious prompts. This approach ensures that the rewritten prompts closely align with the intent of the original prompts while enhancing the attack's effectiveness. Furthermore, we introduce a comprehensive jailbreak evaluation framework incorporating keywords, intent matching, and answer validation to provide a more rigorous and holistic assessment of jailbreak success. Experimental results show the superiority of our approach, achieving state-of-the-art (SOTA) performance on several prominent open and closed-source LLMs, including Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct, and GPT-4o-0806. Our method sets a new benchmark in jailbreak attack effectiveness, highlighting potential vulnerabilities in LLMs. The codebase for this work is available at https://github.com/Aegis1863/xJailbreak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18416v1">Exploring Potential Prompt Injection Attacks in Federated Military LLMs and Their Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      Federated Learning (FL) is increasingly being adopted in military collaborations to develop Large Language Models (LLMs) while preserving data sovereignty. However, prompt injection attacks-malicious manipulations of input prompts-pose new threats that may undermine operational security, disrupt decision-making, and erode trust among allies. This perspective paper highlights four potential vulnerabilities in federated military LLMs: secret data leakage, free-rider exploitation, system disruption, and misinformation spread. To address these potential risks, we propose a human-AI collaborative framework that introduces both technical and policy countermeasures. On the technical side, our framework uses red/blue team wargaming and quality assurance to detect and mitigate adversarial behaviors of shared LLM weights. On the policy side, it promotes joint AI-human policy development and verification of security protocols. Our findings will guide future research and emphasize proactive strategies for emerging military contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01805v2">Locret: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 Preprints
    </div>
    <details class="paper-abstract">
      Scaling the input context length of a large language model (LLM) incurs a significant increase in computation cost and memory footprint to maintain the attention key-value (KV) cache. Existing KV cache compression methods suffer from inefficient compression strategies and limited memory reduction effects, making it difficult for LLMs to conduct long-context inference on consumer-grade devices, especially when inferring long-context stream input. Such obstacles prevent consumer-grade devices from supporting more complex applications, creating challenges for the democratization of LLMs. To overcome this, we propose Locret, the first framework to create an eviction policy compatible with chunked prefill. By evaluating the causal importance of KV cache units by learnable retaining heads, Locret enables precise eviction of cache units, facilitating efficient long-context inference. In our extensive empirical studies, Locret outperforms the recent popular and competitive approaches in terms of memory efficiency and generation quality -- Locret achieves up to 20x of KV cache compression ratio within less than 10% performance loss. Furthermore, Locret achieves 128K+ long-context inference on a single NVIDIA 4090 GPU without compromising generation quality and only costs <1 GPU hour of additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18320v1">Leveraging LLM Agents for Automated Optimization Modeling for SASP Problems: A Graph-RAG based Approach</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      Automated optimization modeling (AOM) has evoked considerable interest with the rapid evolution of large language models (LLMs). Existing approaches predominantly rely on prompt engineering, utilizing meticulously designed expert response chains or structured guidance. However, prompt-based techniques have failed to perform well in the sensor array signal processing (SASP) area due the lack of specific domain knowledge. To address this issue, we propose an automated modeling approach based on retrieval-augmented generation (RAG) technique, which consists of two principal components: a multi-agent (MA) structure and a graph-based RAG (Graph-RAG) process. The MA structure is tailored for the architectural AOM process, with each agent being designed based on principles of human modeling procedure. The Graph-RAG process serves to match user query with specific SASP modeling knowledge, thereby enhancing the modeling result. Results on ten classical signal processing problems demonstrate that the proposed approach (termed as MAG-RAG) outperforms several AOM benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18280v1">Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18265v1">Collecting Cost-Effective, High-Quality Truthfulness Assessments with LLM Summarized Evidence</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 18 pages; 7 figures; 5 tables
    </div>
    <details class="paper-abstract">
      With the degradation of guardrails against mis- and disinformation online, it is more critical than ever to be able to effectively combat it. In this paper, we explore the efficiency and effectiveness of using crowd-sourced truthfulness assessments based on condensed, large language model (LLM) generated summaries of online sources. We compare the use of generated summaries to the use of original web pages in an A/B testing setting, where we employ a large and diverse pool of crowd-workers to perform the truthfulness assessment. We evaluate the quality of assessments, the efficiency with which assessments are performed, and the behavior and engagement of participants. Our results demonstrate that the Summary modality, which relies on summarized evidence, offers no significant change in assessment accuracy over the Standard modality, while significantly increasing the speed with which assessments are performed. Workers using summarized evidence produce a significantly higher number of assessments in the same time frame, reducing the cost needed to acquire truthfulness assessments. Additionally, the Summary modality maximizes both the inter-annotator agreements as well as the reliance on and perceived usefulness of evidence, demonstrating the utility of summarized evidence without sacrificing the quality of assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18243v1">Statistical multi-metric evaluation and visualization of LLM system predictive performance</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      The evaluation of generative or discriminative large language model (LLM)-based systems is often a complex multi-dimensional problem. Typically, a set of system configuration alternatives are evaluated on one or more benchmark datasets, each with one or more evaluation metrics, which may differ between datasets. We often want to evaluate -- with a statistical measure of significance -- whether systems perform differently either on a given dataset according to a single metric, on aggregate across metrics on a dataset, or across datasets. Such evaluations can be done to support decision-making, such as deciding whether a particular system component change (e.g., choice of LLM or hyperparameter values) significantly improves performance over the current system configuration, or, more generally, whether a fixed set of system configurations (e.g., a leaderboard list) have significantly different performances according to metrics of interest. We present a framework implementation that automatically performs the correct statistical tests, properly aggregates the statistical results across metrics and datasets (a nontrivial task), and can visualize the results. The framework is demonstrated on the multi-lingual code generation benchmark CrossCodeEval, for several state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10979v2">Control LLM: Controlled Evolution for Intelligence Retention in LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demand significant computational resources, making it essential to enhance their capabilities without retraining from scratch. A key challenge in this domain is \textit{catastrophic forgetting} (CF), which hampers performance during Continuous Pre-training (CPT) and Continuous Supervised Fine-Tuning (CSFT). We propose \textbf{Control LLM}, a novel approach that leverages parallel pre-trained and expanded transformer blocks, aligning their hidden-states through interpolation strategies This method effectively preserves performance on existing tasks while seamlessly integrating new knowledge. Extensive experiments demonstrate the effectiveness of Control LLM in both CPT and CSFT. On Llama3.1-8B-Instruct, it achieves significant improvements in mathematical reasoning ($+14.4\%$ on Math-Hard) and coding performance ($+10\%$ on MBPP-PLUS). On Llama3.1-8B, it enhances multilingual capabilities ($+10.6\%$ on C-Eval, $+6.8\%$ on CMMLU, and $+30.2\%$ on CMMLU-0shot-CoT). It surpasses existing methods and achieves SOTA among open-source models tuned from the same base model, using substantially less data and compute. Crucially, these gains are realized while preserving strong original capabilities, with minimal degradation ($<4.3\% \text{on MMLU}$) compared to $>35\%$ in open-source Math and Coding models. This approach has been successfully deployed in LinkedIn's GenAI-powered job seeker and Ads unit products. To support further research, we release the training and evaluation code (https://github.com/linkedin/ControlLLM) along with models trained on public datasets (https://huggingface.co/ControlLLM) to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16513v2">Deception in LLMs: Self-Preservation and Autonomous Goals in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 Corrected Version - Solved Some Issues with reference compilation by latex
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have incorporated planning and reasoning capabilities, enabling models to outline steps before execution and provide transparent reasoning paths. This enhancement has reduced errors in mathematical and logical tasks while improving accuracy. These developments have facilitated LLMs' use as agents that can interact with tools and adapt their responses based on new information. Our study examines DeepSeek R1, a model trained to output reasoning tokens similar to OpenAI's o1. Testing revealed concerning behaviors: the model exhibited deceptive tendencies and demonstrated self-preservation instincts, including attempts of self-replication, despite these traits not being explicitly programmed (or prompted). These findings raise concerns about LLMs potentially masking their true objectives behind a facade of alignment. When integrating such LLMs into robotic systems, the risks become tangible - a physically embodied AI exhibiting deceptive behaviors and self-preservation instincts could pursue its hidden objectives through real-world actions. This highlights the critical need for robust goal specification and safety frameworks before any physical implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16493v3">NoTeeline: Supporting Real-Time, Personalized Notetaking with LLM-Enhanced Micronotes</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 Conditionally accepted to IUI 2025
    </div>
    <details class="paper-abstract">
      Taking notes quickly while effectively capturing key information can be challenging, especially when watching videos that present simultaneous visual and auditory streams. Manually taken notes often miss crucial details due to the fast-paced nature of the content, while automatically generated notes fail to incorporate user preferences and discourage active engagement with the content. To address this, we propose an interactive system, NoTeeline, for supporting real-time, personalized notetaking. Given micronotes, NoTeeline automatically expands them into full-fledged notes using a Large Language Model (LLM). The generated notes build on the content of micronotes by adding relevant details while maintaining consistency with the user's writing style. In a within-subjects study (n=12), we found that NoTeeline creates high-quality notes that capture the essence of participant micronotes with 93.2% factual correctness and accurately align with participant writing style (8.33% improvement). Using NoTeeline, participants could capture their desired notes with significantly reduced mental effort, writing 47.0% less text and completing their notes in 43.9% less time compared to a manual notetaking baseline. Our results suggest that NoTeeline enables users to integrate LLM assistance in a familiar notetaking workflow while ensuring consistency with their preferences - providing an example of how to address broader challenges in designing AI-assisted tools to augment human capabilities without compromising user autonomy and personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17310v2">Probing LLM World Models: Enhancing Guesstimation with Wisdom of Crowds Decoding</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      Guesstimation, the task of making approximate quantity estimates, is a common real-world challenge. However, it has been largely overlooked in large language models (LLMs) and vision language models (VLMs) research. We introduce a novel guesstimation dataset, MARBLES. This dataset requires one to estimate how many items (e.g., marbles) can fit into containers (e.g., a one-cup measuring cup), both with and without accompanying images. Inspired by the social science concept of the ``Wisdom of Crowds'' (WOC) - taking the median from estimates from a crowd), which has proven effective in guesstimation, we propose ``WOC decoding'' strategy for LLM guesstimation. We show that LLMs/VLMs perform well on guesstimation, suggesting that they possess some level of a "world model" necessary for guesstimation. Moreover, similar to human performance, the WOC decoding method improves LLM/VLM guesstimation accuracy. Furthermore, the inclusion of images in the multimodal condition enhances model performance. These results highlight the value of WOC decoding strategy for LLMs/VLMs and position guesstimation as a probe for evaluating LLMs/VLMs' world model. As LLMs' world model is a fundamental prerequisite for many real-world tasks, e.g., human-AI teaming, our findings have broad implications for the AI community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18160v1">RepoAudit: An Autonomous LLM-Agent for Repository-Level Code Auditing</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 19 pages, 8 tables, 5 figures, 3 listings
    </div>
    <details class="paper-abstract">
      Code auditing is a code review process with the goal of finding bugs. Large Language Models (LLMs) have shown substantial potential in this task, offering the ability to analyze programs without compilation and enabling customized bug detection following specified prompts. However, applying LLMs to repository-level code auditing presents notable challenges. The inherent context limits and hallucinations of LLMs can lead to the low quality of bug reports. Meanwhile, the large size of software repositories introduces substantial time and token costs, hindering efficiency and scalability in real-world scenarios. This work introduces an autonomous LLM-agent, RepoAudit, designed to enable precise and efficient repository-level code auditing. Equipped with the agent memory, RepoAudit explores the code repository on demand, analyzing data-flow facts along different feasible program paths in individual functions. It also introduces the validator to check the data-flow facts for hallucination mitigation and examine the satisfiability of path conditions of potential buggy paths, which enables RepoAudit to discard false positives in the code auditing. Our experiment shows that RepoAudit powered by Claude 3.5 Sonnet successfully finds 38 true bugs in 15 real-world systems, consuming 0.44 hours and $2.54 per project on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17186v2">Complete Chess Games Enable LLM Become A Chess Master</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLM) have shown remarkable abilities in text generation, question answering, language translation, reasoning and many other tasks. It continues to advance rapidly and is becoming increasingly influential in various fields, from technology and business to education and entertainment. Despite LLM's success in multiple areas, its ability to play abstract games, such as chess, is underexplored. Chess-playing requires the language models to output legal and reasonable moves from textual inputs. Here, we propose the Large language model ChessLLM to play full chess games. We transform the game into a textual format with the best move represented in the Forsyth-Edwards Notation. We show that by simply supervised fine-tuning, our model has achieved a professional-level Elo rating of 1788 in matches against the standard Elo-rated Stockfish when permitted to sample 10 times. We further show that data quality is important. Long-round data supervision enjoys a 350 Elo rating improvement over short-round data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18123v1">Battery State of Health Estimation Using LLM Framework</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 Accepted at The 26th International Symposium on Quality Electronic Design (ISQED'25)
    </div>
    <details class="paper-abstract">
      Battery health monitoring is critical for the efficient and reliable operation of electric vehicles (EVs). This study introduces a transformer-based framework for estimating the State of Health (SoH) and predicting the Remaining Useful Life (RUL) of lithium titanate (LTO) battery cells by utilizing both cycle-based and instantaneous discharge data. Testing on eight LTO cells under various cycling conditions over 500 cycles, we demonstrate the impact of charge durations on energy storage trends and apply Differential Voltage Analysis (DVA) to monitor capacity changes (dQ/dV) across voltage ranges. Our LLM model achieves superior performance, with a Mean Absolute Error (MAE) as low as 0.87\% and varied latency metrics that support efficient processing, demonstrating its strong potential for real-time integration into EVs. The framework effectively identifies early signs of degradation through anomaly detection in high-resolution data, facilitating predictive maintenance to prevent sudden battery failures and enhance energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18099v1">Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-01-30
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge models generate chain-of-thought (CoT) sequences intended to capture the step-bystep reasoning process that underlies the final evaluation of a response. However, due to the lack of human annotated CoTs for evaluation, the required components and structure of effective reasoning traces remain understudied. Consequently, previous approaches often (1) constrain reasoning traces to hand-designed components, such as a list of criteria, reference answers, or verification questions and (2) structure them such that planning is intertwined with the reasoning for evaluation. In this work, we propose EvalPlanner, a preference optimization algorithm for Thinking-LLM-as-a-Judge that first generates an unconstrained evaluation plan, followed by its execution, and then the final judgment. In a self-training loop, EvalPlanner iteratively optimizes over synthetically constructed evaluation plans and executions, leading to better final verdicts. Our method achieves a new state-of-the-art performance for generative reward models on RewardBench (with a score of 93.9), despite being trained on fewer amount of, and synthetically generated, preference pairs. Additional experiments on other benchmarks like RM-Bench, JudgeBench, and FollowBenchEval further highlight the utility of both planning and reasoning for building robust LLM-as-a-Judge reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18096v1">LLMs can see and hear without any training</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 Code: https://github.com/facebookresearch/MILS
    </div>
    <details class="paper-abstract">
      We present MILS: Multimodal Iterative LLM Solver, a surprisingly simple, training-free approach, to imbue multimodal capabilities into your favorite LLM. Leveraging their innate ability to perform multi-step reasoning, MILS prompts the LLM to generate candidate outputs, each of which are scored and fed back iteratively, eventually generating a solution to the task. This enables various applications that typically require training specialized models on task-specific data. In particular, we establish a new state-of-the-art on emergent zero-shot image, video and audio captioning. MILS seamlessly applies to media generation as well, discovering prompt rewrites to improve text-to-image generation, and even edit prompts for style transfer! Finally, being a gradient-free optimization approach, MILS can invert multimodal embeddings into text, enabling applications like cross-modal arithmetic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06711v2">MathVC: An LLM-Simulated Multi-Character Virtual Classroom for Mathematics Education</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 Accepted by AAAI workshop
    </div>
    <details class="paper-abstract">
      Mathematical modeling (MM) is considered a fundamental skill for students in STEM disciplines. Practicing the MM skill is often the most effective when students can engage in group discussion and collaborative problem-solving. However, due to unevenly distributed teachers and educational resources needed to monitor such group activities, students do not always receive equal opportunities for this practice. Excitingly, large language models (LLMs) have recently demonstrated strong capability in both modeling mathematical problems and simulating characters with different traits and properties. Drawing inspiration from the advancement of LLMs, in this work, we present MATHVC, the very first LLM-powered virtual classroom containing multiple LLM-simulated student characters, with whom a human student can practice their MM skill. To encourage each LLM character's behaviors to be aligned with their specified math-relevant properties (termed "characteristics alignment") and the overall conversational procedure to be close to an authentic student MM discussion (termed "conversational procedural alignment"), we proposed three innovations: integrating MM domain knowledge into the simulation, defining a symbolic schema as the ground for character simulation, and designing a meta planner at the platform level to drive the conversational procedure. Through experiments and ablation studies, we confirmed the effectiveness of our simulation approach and showed the promise for MATHVC to benefit real-life students in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18056v1">RL-based Query Rewriting with Distilled LLM for online E-Commerce Systems</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      Query rewriting (QR) is a critical technique in e-commerce search, addressing the lexical gap between user queries and product descriptions to enhance search performance. Existing QR approaches typically fall into two categories: discriminative models and generative methods leveraging large language models (LLMs). Discriminative models often struggle with natural language understanding and offer limited flexibility in rewriting, while generative LLMs, despite producing high-quality rewrites, face high inference latency and cost in online settings. These limitations force offline deployment, making them vulnerable to issues like information staleness and semantic drift. To overcome these challenges, we propose a novel hybrid pipeline for QR that balances efficiency and effectiveness. Our approach combines offline knowledge distillation to create a lightweight but efficient student model with online reinforcement learning (RL) to refine query rewriting dynamically using real-time feedback. A key innovation is the use of LLMs as simulated human feedback, enabling scalable reward signals and cost-effective evaluation without manual annotations. Experimental results on Amazon ESCI dataset demonstrate significant improvements in query relevance, diversity, and adaptability, as well as positive feedback from the LLM simulation. This work contributes to advancing LLM capabilities for domain-specific applications, offering a robust solution for dynamic and complex e-commerce search environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14575v4">EVINCE: Optimizing Multi-LLM Dialogues Using Conditional Statistics and Information Theory</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 24 pages, 9 figures, 10 tables. arXiv admin note: text overlap with arXiv:2405.15808
    </div>
    <details class="paper-abstract">
      EVINCE (Entropy and Variation IN Conditional Exchanges) is a novel framework for optimizing multi-LLM dialogues using conditional statistics and information theory. It addresses limitations in multi-agent debate (MAS) frameworks, where multiple LLMs ``chat'' without behavior modulation or mutual information quality assessment. Using dual entropy optimization to balance perspective diversity and prior knowledge, $\EVINCE$ provides quantitative tools to dynamically regulate LLM linguistic behaviors. When mutual information is low and both cross-entropy and Wasserstein distance are high, EVINCE promotes contentious dialogues to expose diverse perspectives and uncover inconsistencies. Conversely, as cross-entropy decreases and mutual information stabilizes, it transitions discussions into a conciliatory phase, encouraging compromise and acknowledgment of valid points. Using information-theoretic metrics and optimizing mutual information, $\EVINCE$ emerges as a structured and highly effective framework for multi-LLM collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17981v1">Can Generative LLMs Create Query Variants for Test Collections? An Exploratory Study</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 Published in the proceedings of SIGIR'23
    </div>
    <details class="paper-abstract">
      This paper explores the utility of a Large Language Model (LLM) to automatically generate queries and query variants from a description of an information need. Given a set of information needs described as backstories, we explore how similar the queries generated by the LLM are to those generated by humans. We quantify the similarity using different metrics and examine how the use of each set would contribute to document pooling when building test collections. Our results show potential in using LLMs to generate query variants. While they may not fully capture the wide variety of human-generated variants, they generate similar sets of relevant documents, reaching up to 71.1% overlap at a pool depth of 100.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17969v1">LLMs can be Fooled into Labelling a Document as Relevant (best café near me; this paper is perfectly relevant)</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 Published in the proceedings of SIGIR-AP'24
    </div>
    <details class="paper-abstract">
      LLMs are increasingly being used to assess the relevance of information objects. This work reports on experiments to study the labelling of short texts (i.e., passages) for relevance, using multiple open-source and proprietary LLMs. While the overall agreement of some LLMs with human judgements is comparable to human-to-human agreement measured in previous research, LLMs are more likely to label passages as relevant compared to human judges, indicating that LLM labels denoting non-relevance are more reliable than those indicating relevance. This observation prompts us to further examine cases where human judges and LLMs disagree, particularly when the human judge labels the passage as non-relevant and the LLM labels it as relevant. Results show a tendency for many LLMs to label passages that include the original query terms as relevant. We, therefore, conduct experiments to inject query words into random and irrelevant passages, not unlike the way we inserted the query "best caf\'e near me" into this paper. The results show that LLMs are highly influenced by the presence of query words in the passages under assessment, even if the wider passage has no relevance to the query. This tendency of LLMs to be fooled by the mere presence of query words demonstrates a weakness in our current measures of LLM labelling: relying on overall agreement misses important patterns of failures. There is a real risk of bias in LLM-generated relevance labels and, therefore, a risk of bias in rankers trained on those labels. We also investigate the effects of deliberately manipulating LLMs by instructing them to label passages as relevant, similar to the instruction "this paper is perfectly relevant" inserted above. We find that such manipulation influences the performance of some LLMs, highlighting the critical need to consider potential vulnerabilities when deploying LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17942v1">"I Would Never Trust Anything Western": Kumu (Educator) Perspectives on Use of LLMs for Culturally Revitalizing CS Education in Hawaiian Schools</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into educational technology, their potential to assist in developing curricula has gained interest among educators. Despite this growing attention, their applicability in culturally responsive Indigenous educational settings like Hawai`i's public schools and Kaiapuni (immersion language) programs, remains understudied. Additionally, `Olelo Hawai`i, the Hawaiian language, as a low-resource language, poses unique challenges and concerns about cultural sensitivity and the reliability of generated content. Through surveys and interviews with kumu (educators), this study explores the perceived benefits and limitations of using LLMs for culturally revitalizing computer science (CS) education in Hawaiian public schools with Kaiapuni programs. Our findings highlight AI's time-saving advantages while exposing challenges such as cultural misalignment and reliability concerns. We conclude with design recommendations for future AI tools to better align with Hawaiian cultural values and pedagogical practices, towards the broader goal of trustworthy, effective, and culturally grounded AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17840v1">Learning Beyond the Surface: How Far Can Continual Pre-Training with LoRA Enhance LLMs' Domain-Specific Insight Learning?</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance on various tasks, yet their ability to extract and internalize deeper insights from domain-specific datasets remains underexplored. In this study, we investigate how continual pre-training can enhance LLMs' capacity for insight learning across three distinct forms: declarative, statistical, and probabilistic insights. Focusing on two critical domains: medicine and finance, we employ LoRA to train LLMs on two existing datasets. To evaluate each insight type, we create benchmarks to measure how well continual pre-training helps models go beyond surface-level knowledge. We also assess the impact of document modification on capturing insights. The results show that, while continual pre-training on original documents has a marginal effect, modifying documents to retain only essential information significantly enhances the insight-learning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17785v1">Reasoning Over the Glyphs: Evaluation of LLM's Decipherment of Rare Scripts</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 7 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We explore the capabilities of LVLMs and LLMs in deciphering rare scripts not encoded in Unicode. We introduce a novel approach to construct a multimodal dataset of linguistic puzzles involving such scripts, utilizing a tokenization method for language glyphs. Our methods include the Picture Method for LVLMs and the Description Method for LLMs, enabling these models to tackle these challenges. We conduct experiments using prominent models, GPT-4o, Gemini, and Claude 3.5 Sonnet, on linguistic puzzles. Our findings reveal the strengths and limitations of current AI methods in linguistic decipherment, highlighting the impact of Unicode encoding on model performance and the challenges of modeling visual language tokens through descriptions. Our study advances understanding of AI's potential in linguistic decipherment and underscores the need for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17771v1">2SSP: A Two-Stage Framework for Structured Pruning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      We propose a novel Two-Stage framework for Structured Pruning (2SSP) for pruning Large Language Models (LLMs), which combines two different strategies of pruning, namely Width and Depth Pruning. The first stage (Width Pruning) removes entire neurons, hence their corresponding rows and columns, aiming to preserve the connectivity among the pruned structures in the intermediate state of the Feed-Forward Networks in each Transformer block. This is done based on an importance score measuring the impact of each neuron over the output magnitude. The second stage (Depth Pruning), instead, removes entire Attention submodules. This is done by applying an iterative process that removes the Attention submodules with the minimum impact on a given metric of interest (in our case, perplexity). We also propose a novel mechanism to balance the sparsity rate of the two stages w.r.t. to the desired global sparsity. We test 2SSP on four LLM families and three sparsity rates (25\%, 37.5\%, and 50\%), measuring the resulting perplexity over three language modeling datasets as well as the performance over six downstream tasks. Our method consistently outperforms five state-of-the-art competitors over three language modeling and six downstream tasks, with an up to two-order-of-magnitude gain in terms of pruning time. The code is available at available at \url{https://github.com/FabrizioSandri/2SSP}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17767v1">Hybrid Graphs for Table-and-Text based Question Answering using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 Accepted at NAACL 2025 Main Track
    </div>
    <details class="paper-abstract">
      Answering questions that require reasoning and aggregation across both structured (tables) and unstructured (raw text) data sources presents significant challenges. Current methods rely on fine-tuning and high-quality, human-curated data, which is difficult to obtain. Recent advances in Large Language Models (LLMs) have shown promising results for multi-hop question answering (QA) over single-source text data in a zero-shot setting, yet exploration into multi-source Table-Text QA remains limited. In this paper, we present a novel Hybrid Graph-based approach for Table-Text QA that leverages LLMs without fine-tuning. Our method constructs a unified Hybrid Graph from textual and tabular data, pruning information based on the input question to provide the LLM with relevant context concisely. We evaluate our approach on the challenging Hybrid-QA and OTT-QA datasets using state-of-the-art LLMs, including GPT-3.5, GPT-4, and LLaMA-3. Our method achieves the best zero-shot performance on both datasets, improving Exact Match scores by up to 10% on Hybrid-QA and 5.4% on OTT-QA. Moreover, our approach reduces token usage by up to 53% compared to the original context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12112v2">Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 57 pages, 25 figures, 15 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have recently demonstrated strong potential in solving planning problems, there is a trade-off between flexibility and complexity. LLMs, as zero-shot planners themselves, are still not capable of directly generating valid plans for complex planning problems such as multi-constraint or long-horizon tasks. On the other hand, many frameworks aiming to solve complex planning problems often rely on task-specific preparatory efforts, such as task-specific in-context examples and pre-defined critics/verifiers, which limits their cross-task generalization capability. In this paper, we tackle these challenges by observing that the core of many planning problems lies in optimization problems: searching for the optimal solution (best plan) with goals subject to constraints (preconditions and effects of decisions). With LLMs' commonsense, reasoning, and programming capabilities, this opens up the possibilities of a universal LLM-based approach to planning problems. Inspired by this observation, we propose LLMFP, a general-purpose framework that leverages LLMs to capture key information from planning problems and formally formulate and solve them as optimization problems from scratch, with no task-specific examples needed. We apply LLMFP to 9 planning problems, ranging from multi-constraint decision making to multi-step planning problems, and demonstrate that LLMFP achieves on average 83.7% and 86.8% optimal rate across 9 tasks for GPT-4o and Claude 3.5 Sonnet, significantly outperforming the best baseline (direct planning with OpenAI o1-preview) with 37.6% and 40.7% improvements. We also validate components of LLMFP with ablation experiments and analyzed the underlying success and failure reasons. Project page: https://sites.google.com/view/llmfp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16135v3">Can LLMs Obfuscate Code? A Systematic Analysis of Large Language Models into Assembly Code Obfuscation</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 To appear in AAAI 2025, Main Track
    </div>
    <details class="paper-abstract">
      Malware authors often employ code obfuscations to make their malware harder to detect. Existing tools for generating obfuscated code often require access to the original source code (e.g., C++ or Java), and adding new obfuscations is a non-trivial, labor-intensive process. In this study, we ask the following question: Can Large Language Models (LLMs) potentially generate a new obfuscated assembly code? If so, this poses a risk to anti-virus engines and potentially increases the flexibility of attackers to create new obfuscation patterns. We answer this in the affirmative by developing the MetamorphASM benchmark comprising MetamorphASM Dataset (MAD) along with three code obfuscation techniques: dead code, register substitution, and control flow change. The MetamorphASM systematically evaluates the ability of LLMs to generate and analyze obfuscated code using MAD, which contains 328,200 obfuscated assembly code samples. We release this dataset and analyze the success rate of various LLMs (e.g., GPT-3.5/4, GPT-4o-mini, Starcoder, CodeGemma, CodeLlama, CodeT5, and LLaMA 3.1) in generating obfuscated assembly code. The evaluation was performed using established information-theoretic metrics and manual human review to ensure correctness and provide the foundation for researchers to study and develop remediations to this risk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17630v1">Uncertainty Quantification and Decomposition for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 WWW 2025
    </div>
    <details class="paper-abstract">
      Despite the widespread adoption of large language models (LLMs) for recommendation, we demonstrate that LLMs often exhibit uncertainty in their recommendations. To ensure the trustworthy use of LLMs in generating recommendations, we emphasize the importance of assessing the reliability of recommendations generated by LLMs. We start by introducing a novel framework for estimating the predictive uncertainty to quantitatively measure the reliability of LLM-based recommendations. We further propose to decompose the predictive uncertainty into recommendation uncertainty and prompt uncertainty, enabling in-depth analyses of the primary source of uncertainty. Through extensive experiments, we (1) demonstrate predictive uncertainty effectively indicates the reliability of LLM-based recommendations, (2) investigate the origins of uncertainty with decomposed uncertainty measures, and (3) propose uncertainty-aware prompting for a lower predictive uncertainty and enhanced recommendation. Our source code and model weights are available at https://github.com/WonbinKweon/UNC_LLM_REC_WWW2025
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03664v3">LLMs in the Heart of Differential Testing: A Case Study on a Medical Rule Engine</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 12 pages, 6 figures, 4 tables, 1 listing, revised arguments
    </div>
    <details class="paper-abstract">
      The Cancer Registry of Norway (CRN) uses an automated cancer registration support system (CaReSS) to support core cancer registry activities, i.e, data capture, data curation, and producing data products and statistics for various stakeholders. GURI is a core component of CaReSS, which is responsible for validating incoming data with medical rules. Such medical rules are manually implemented by medical experts based on medical standards, regulations, and research. Since large language models (LLMs) have been trained on a large amount of public information, including these documents, they can be employed to generate tests for GURI. Thus, we propose an LLM-based test generation and differential testing approach (LLMeDiff) to test GURI. We experimented with four different LLMs, two medical rule engine implementations, and 58 real medical rules to investigate the hallucination, success, time efficiency, and robustness of the LLMs to generate tests, and these tests' ability to find potential issues in GURI. Our results showed that GPT-3.5 hallucinates the least, is the most successful, and is generally the most robust; however, it has the worst time efficiency. Our differential testing revealed 22 medical rules where implementation inconsistencies were discovered (e.g., regarding handling rule versions). Finally, we provide insights for practitioners and researchers based on the results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14326v2">medIKAL: Integrating Knowledge Graphs as Assistants of LLMs for Enhanced Clinical Diagnosis on EMRs</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      Electronic Medical Records (EMRs), while integral to modern healthcare, present challenges for clinical reasoning and diagnosis due to their complexity and information redundancy. To address this, we proposed medIKAL (Integrating Knowledge Graphs as Assistants of LLMs), a framework that combines Large Language Models (LLMs) with knowledge graphs (KGs) to enhance diagnostic capabilities. medIKAL assigns weighted importance to entities in medical records based on their type, enabling precise localization of candidate diseases within KGs. It innovatively employs a residual network-like approach, allowing initial diagnosis by the LLM to be merged into KG search results. Through a path-based reranking algorithm and a fill-in-the-blank style prompt template, it further refined the diagnostic process. We validated medIKAL's effectiveness through extensive experiments on a newly introduced open-sourced Chinese EMR dataset, demonstrating its potential to improve clinical diagnosis in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17510v1">LLM Assistance for Pediatric Depression</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      Traditional depression screening methods, such as the PHQ-9, are particularly challenging for children in pediatric primary care due to practical limitations. AI has the potential to help, but the scarcity of annotated datasets in mental health, combined with the computational costs of training, highlights the need for efficient, zero-shot approaches. In this work, we investigate the feasibility of state-of-the-art LLMs for depressive symptom extraction in pediatric settings (ages 6-24). This approach aims to complement traditional screening and minimize diagnostic errors. Our findings show that all LLMs are 60% more efficient than word match, with Flan leading in precision (average F1: 0.65, precision: 0.78), excelling in the extraction of more rare symptoms like "sleep problems" (F1: 0.92) and "self-loathing" (F1: 0.8). Phi strikes a balance between precision (0.44) and recall (0.60), performing well in categories like "Feeling depressed" (0.69) and "Weight change" (0.78). Llama 3, with the highest recall (0.90), overgeneralizes symptoms, making it less suitable for this type of analysis. Challenges include the complexity of clinical notes and overgeneralization from PHQ-9 scores. The main challenges faced by LLMs include navigating the complex structure of clinical notes with content from different times in the patient trajectory, as well as misinterpreting elevated PHQ-9 scores. We finally demonstrate the utility of symptom annotations provided by Flan as features in an ML algorithm, which differentiates depression cases from controls with high precision of 0.78, showing a major performance boost compared to a baseline that does not use these features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17479v1">DFPE: A Diverse Fingerprint Ensemble for Enhancing LLM Performance</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities across various natural language processing tasks but often struggle to excel uniformly in diverse or complex domains. We propose a novel ensemble method - Diverse Fingerprint Ensemble (DFPE), which leverages the complementary strengths of multiple LLMs to achieve more robust performance. Our approach involves: (1) clustering models based on response "fingerprints" patterns, (2) applying a quantile-based filtering mechanism to remove underperforming models at a per-subject level, and (3) assigning adaptive weights to remaining models based on their subject-wise validation accuracy. In experiments on the Massive Multitask Language Understanding (MMLU) benchmark, DFPE outperforms the best single model by 3% overall accuracy and 5% in discipline-level accuracy. This method increases the robustness and generalization of LLMs and underscores how model selection, diversity preservation, and performance-driven weighting can effectively address challenging, multi-faceted language understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17461v1">AugmenTest: Enhancing Tests with LLM-Driven Oracles</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 Accepted for publication at 2025 IEEE Conference on Software Testing, Verification and Validation (ICST)
    </div>
    <details class="paper-abstract">
      Automated test generation is crucial for ensuring the reliability and robustness of software applications while at the same time reducing the effort needed. While significant progress has been made in test generation research, generating valid test oracles still remains an open problem. To address this challenge, we present AugmenTest, an approach leveraging Large Language Models (LLMs) to infer correct test oracles based on available documentation of the software under test. Unlike most existing methods that rely on code, AugmenTest utilizes the semantic capabilities of LLMs to infer the intended behavior of a method from documentation and developer comments, without looking at the code. AugmenTest includes four variants: Simple Prompt, Extended Prompt, RAG with a generic prompt (without the context of class or method under test), and RAG with Simple Prompt, each offering different levels of contextual information to the LLMs. To evaluate our work, we selected 142 Java classes and generated multiple mutants for each. We then generated tests from these mutants, focusing only on tests that passed on the mutant but failed on the original class, to ensure that the tests effectively captured bugs. This resulted in 203 unique tests with distinct bugs, which were then used to evaluate AugmenTest. Results show that in the most conservative scenario, AugmenTest's Extended Prompt consistently outperformed the Simple Prompt, achieving a success rate of 30\% for generating correct assertions. In comparison, the state-of-the-art TOGA approach achieved 8.2\%. Contrary to our expectations, the RAG-based approaches did not lead to improvements, with performance of 18.2\% success rate for the most conservative scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04437v3">vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 To appear in ASPLOS 2025
    </div>
    <details class="paper-abstract">
      PagedAttention is a popular approach for dynamic memory allocation in LLM serving systems. It enables on-demand allocation of GPU memory to mitigate KV cache fragmentation -- a phenomenon that crippled the batch size (and consequently throughput) in prior systems. However, in trying to allocate physical memory at runtime, PagedAttention ends up changing the virtual memory layout of the KV cache from contiguous to non-contiguous. Such a design leads to non-trivial programming and performance overheads. We present vAttention -- an approach that mitigates fragmentation in physical memory while retaining the contiguity of KV cache in virtual memory. We achieve this by decoupling the allocation of virtual and physical memory using CUDA virtual memory management APIs. We also introduce various LLM-specific optimizations to address the limitations of CUDA virtual memory support. Overall, vAttention is a simpler, portable, and performant alternative to PagedAttention: it supports various attention kernels out-of-the-box and improves LLM serving throughput by up to 1.23x compared to the use of PagedAttention-based kernels of FlashAttention and FlashInfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17399v1">MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      We present MultiChallenge, a pioneering benchmark evaluating large language models (LLMs) on conducting multi-turn conversations with human users, a crucial yet underexamined capability for their applications. MultiChallenge identifies four categories of challenges in multi-turn conversations that are not only common and realistic among current human-LLM interactions, but are also challenging to all current frontier LLMs. All 4 challenges require accurate instruction-following, context allocation, and in-context reasoning at the same time. We also develop LLM as judge with instance-level rubrics to facilitate an automatic evaluation method with fair agreement with experienced human raters. Despite achieving near-perfect scores on existing multi-turn evaluation benchmarks, all frontier models have less than 50% accuracy on MultiChallenge, with the top-performing Claude 3.5 Sonnet (June 2024) achieving just a 41.4% average accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06647v3">How Efficient is LLM-Generated Code? A Rigorous & High-Standard Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has significantly pushed the frontiers of program synthesis. Advancement of LLM-based program synthesis calls for a thorough evaluation of LLM-generated code. Most evaluation frameworks focus on the (functional) correctness of generated code; efficiency, as an important measure of code quality, has been overlooked in existing evaluations. In this work, we develop ENAMEL (EfficeNcy AutoMatic EvaLuator), a rigorous and high-standard benchmark for evaluating the capability of LLMs in generating efficient code. Firstly, we propose a new efficiency metric called eff@k, which generalizes the pass@k metric from correctness to efficiency and appropriately handles right-censored execution time. Furthermore, we derive an unbiased and variance-reduced estimator of eff@k via Rao--Blackwellization; we also provide a numerically stable implementation for the new estimator. Secondly, to set a high-standard for efficiency evaluation, we employ a human expert to design best algorithms and implementations as our reference solutions of efficiency, many of which are much more efficient than existing canonical solutions in HumanEval and HumanEval+. Moreover, to ensure a rigorous evaluation, we employ a human expert to curate strong test case generators to filter out wrong code and differentiate suboptimal algorithms. An extensive study across 30 popular LLMs using our benchmark ENAMEL shows that LLMs still fall short of generating expert-level efficient code. Using two subsets of our problem set, we demonstrate that such deficiency is because current LLMs struggle in designing advanced algorithms and are barely aware of implementation optimization. Our benchmark is publicly available at https://github.com/q-rz/enamel .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17330v1">Attribution analysis of legal language as used by LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 9 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Three publicly-available LLM specifically designed for legal tasks have been implemented and shown that classification accuracy can benefit from training over legal corpora, but why and how? Here we use two publicly-available legal datasets, a simpler binary classification task of ``overruling'' texts, and a more elaborate multiple choice task identifying ``holding'' judicial decisions. We report on experiments contrasting the legal LLM and a generic BERT model for comparison, against both datasets. We use integrated gradient attribution techniques to impute ``causes'' of variation in the models' perfomance, and characterize them in terms of the tokenizations each use. We find that while all models can correctly classify some test examples from the casehold task, other examples can only be identified by only one, model, and attribution can be used to highlight the reasons for this. We find that differential behavior of the models' tokenizers accounts for most of the difference and analyze these differences in terms of the legal language they process. Frequency analysis of tokens generated by dataset texts, combined with use of known ``stop word'' lists, allow identification of tokens that are clear signifiers of legal topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17273v1">Tailored Truths: Optimizing LLM Persuasion with Personalization and Fabricated Statistics</a></div>
    <div class="paper-meta">
      📅 2025-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly persuasive, demonstrating the ability to personalize arguments in conversation with humans by leveraging their personal data. This may have serious impacts on the scale and effectiveness of disinformation campaigns. We studied the persuasiveness of LLMs in a debate setting by having humans $(n=33)$ engage with LLM-generated arguments intended to change the human's opinion. We quantified the LLM's effect by measuring human agreement with the debate's hypothesis pre- and post-debate and analyzing both the magnitude of opinion change, as well as the likelihood of an update in the LLM's direction. We compare persuasiveness across established persuasion strategies, including personalized arguments informed by user demographics and personality, appeal to fabricated statistics, and a mixed strategy utilizing both personalized arguments and fabricated statistics. We found that static arguments generated by humans and GPT-4o-mini have comparable persuasive power. However, the LLM outperformed static human-written arguments when leveraging the mixed strategy in an interactive debate setting. This approach had a $\mathbf{51\%}$ chance of persuading participants to modify their initial position, compared to $\mathbf{32\%}$ for the static human-written arguments. Our results highlight the concerning potential for LLMs to enable inexpensive and persuasive large-scale disinformation campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17084v1">Token-by-Token Regeneration and Domain Biases: A Benchmark of LLMs on Advanced Mathematical Problem-Solving</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 8 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in many natural language tasks, yet they struggle with complex mathemat-ical problem-solving, particularly in symbolic reasoning and maintaining consistent output. This study evalu-ates 10 LLMs with 7 to 8 billion parameters using 945 competition-level problems from the MATH dataset. The focus is on their ability to generate executable Python code as a step in their reasoning process, involving over 9,450 code executions. The research introduces an evaluation framework using mistral-large-2411 to rate answers on a 5-point scale, which helps address inconsistencies in mathematical notation. It also examines the impact of regenerating output token-by-token on refining results. The findings reveal a significant 34.5% per-formance gap between the top commercial model (gpt-4o-mini, scoring 83.7%) and the least effective open-source model (open-codestral-mamba:v0.1, scoring 49.2%). This disparity is especially noticeable in complex areas like Number Theory. While token-by-token regeneration slightly improved accuracy (+0.8%) for the model llama3.1:8b, it also reduced code execution time by 36.7%, highlighting a trade-off between efficiency and precision. The study also noted a consistent trend where harder problems correlated with lower accuracy across all models. Despite using controlled execution environments, less than 1% of the generated code was unsafe, and 3.17% of problems remained unsolved after 10 attempts, suggesting that hybrid reasoning methods may be beneficial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17032v2">MINTQA: A Multi-Hop Question Answering Benchmark for Evaluating LLMs on New and Tail Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-01-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in various reasoning tasks but face significant challenges with complex, knowledge-intensive multi-hop queries, particularly those involving new or long-tail knowledge. Existing benchmarks often fail to fully address these challenges. To bridge this gap, we introduce MINTQA (Multi-hop Question Answering on New and Tail Knowledge), a comprehensive benchmark to evaluate LLMs' capabilities in multi-hop reasoning across four critical dimensions: question handling strategy, sub-question generation, retrieval-augmented generation, and iterative or dynamic decomposition and retrieval. MINTQA comprises 10,479 question-answer pairs for evaluating new knowledge and 17,887 pairs for assessing long-tail knowledge, with each question equipped with corresponding sub-questions and answers. Our systematic evaluation of 22 state-of-the-art LLMs on MINTQA reveals significant limitations in their ability to handle complex knowledge base queries, particularly in handling new or unpopular knowledge. Our findings highlight critical challenges and offer insights for advancing multi-hop reasoning capabilities. The MINTQA benchmark is available at https://github.com/probe2/multi-hop/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17024v1">Automated Refactoring of Non-Idiomatic Python Code: A Differentiated Replication with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-28
    </div>
    <details class="paper-abstract">
      In the Python ecosystem, the adoption of idiomatic constructs has been fostered because of their expressiveness, increasing productivity and even efficiency, despite controversial arguments concerning familiarity or understandability issues. Recent research contributions have proposed approaches -- based on static code analysis and transformation -- to automatically identify and enact refactoring opportunities of non-idiomatic code into idiomatic ones. Given the potential recently offered by Large Language Models (LLMs) for code-related tasks, in this paper, we present the results of a replication study in which we investigate GPT-4 effectiveness in recommending and suggesting idiomatic refactoring actions. Our results reveal that GPT-4 not only identifies idiomatic constructs effectively but frequently exceeds the benchmark in proposing refactoring actions where the existing baseline failed. A manual analysis of a random sample shows the correctness of the obtained recommendations. Our findings underscore the potential of LLMs to achieve tasks where, in the past, implementing recommenders based on complex code analyses was required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16945v1">ToolFactory: Automating Tool Generation by Leveraging LLM to Understand REST API Documentations</a></div>
    <div class="paper-meta">
      📅 2025-01-28
    </div>
    <details class="paper-abstract">
      LLM-based tool agents offer natural language interfaces, enabling users to seamlessly interact with computing services. While REST APIs are valuable resources for building such agents, they must first be transformed into AI-compatible tools. Automatically generating AI-compatible tools from REST API documents can greatly streamline tool agent development and minimize user learning curves. However, API documentation often suffers from a lack of standardization, inconsistent schemas, and incomplete information. To address these issues, we developed \textbf{ToolFactory}, an open-source pipeline for automating tool generation from unstructured API documents. To enhance the reliability of the developed tools, we implemented an evaluation method to diagnose errors. Furthermore, we built a knowledge base of verified tools, which we leveraged to infer missing information from poorly documented APIs. We developed the API Extraction Benchmark, comprising 167 API documents and 744 endpoints in various formats, and designed a JSON schema to annotate them. This annotated dataset was utilized to train and validate ToolFactory. The experimental results highlight the effectiveness of ToolFactory. We also demonstrated ToolFactory by creating a domain-specific AI agent for glycomaterials research. ToolFactory exhibits significant potential for facilitating the seamless integration of scientific REST APIs into AI workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16899v1">RDMM: Fine-Tuned LLM Models for On-Device Robotic Decision Making with Enhanced Contextual Awareness in Specific Domains</a></div>
    <div class="paper-meta">
      📅 2025-01-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) represent a significant advancement in integrating physical robots with AI-driven systems. We showcase the capabilities of our framework within the context of the real-world household competition. This research introduces a framework that utilizes RDMM (Robotics Decision-Making Models), which possess the capacity for decision-making within domain-specific contexts, as well as an awareness of their personal knowledge and capabilities. The framework leverages information to enhance the autonomous decision-making of the system. In contrast to other approaches, our focus is on real-time, on-device solutions, successfully operating on hardware with as little as 8GB of memory. Our framework incorporates visual perception models equipping robots with understanding of their environment. Additionally, the framework has integrated real-time speech recognition capabilities, thus enhancing the human-robot interaction experience. Experimental results demonstrate that the RDMM framework can plan with an 93\% accuracy. Furthermore, we introduce a new dataset consisting of 27k planning instances, as well as 1.3k text-image annotated samples derived from the competition. The framework, benchmarks, datasets, and models developed in this work are publicly available on our GitHub repository at https://github.com/shadynasrat/RDMM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16865v1">JRE-L: Journalist, Reader, and Editor LLMs in the Loop for Science Journalism for the General Audience</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 arXiv admin note: substantial text overlap with arXiv:2407.09756
    </div>
    <details class="paper-abstract">
      Science journalism reports current scientific discoveries to non-specialists, aiming to enable public comprehension of the state of the art. This task is challenging as the audience often lacks specific knowledge about the presented research. We propose a JRE-L framework that integrates three LLMs mimicking the writing-reading-feedback-revision loop. In JRE-L, one LLM acts as the journalist, another LLM as the general public reader, and the third LLM as an editor. The journalist's writing is iteratively refined by feedback from the reader and suggestions from the editor. Our experiments demonstrate that by leveraging the collaboration of two 7B and one 1.8B open-source LLMs, we can generate articles that are more accessible than those generated by existing methods, including prompting single advanced models such as GPT-4 and other LLM-collaboration strategies. Our code is publicly available at github.com/Zzoay/JRE-L.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16857v1">Comparing Human and LLM Generated Code: The Jury is Still Out!</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Much is promised in relation to AI-supported software development. However, there has been limited evaluation effort in the research domain aimed at validating the true utility of such techniques, especially when compared to human coding outputs. We bridge this gap, where a benchmark dataset comprising 72 distinct software engineering tasks is used to compare the effectiveness of large language models (LLMs) and human programmers in producing Python software code. GPT-4 is used as a representative LLM, where for the code generated by humans and this LLM, we evaluate code quality and adherence to Python coding standards, code security and vulnerabilities, code complexity and functional correctness. We use various static analysis benchmarks, including Pylint, Radon, Bandit and test cases. Among the notable outcomes, results show that human-generated code recorded higher ratings for adhering to coding standards than GPT-4. We observe security flaws in code generated by both humans and GPT-4, however, code generated by humans shows a greater variety of problems, but GPT-4 code included more severe outliers. Our results show that although GPT-4 is capable of producing coding solutions, it frequently produces more complex code that may need more reworking to ensure maintainability. On the contrary however, our outcomes show that a higher number of test cases passed for code generated by GPT-4 across a range of tasks than code that was generated by humans. That said, GPT-4 frequently struggles with complex problem-solving that involve in-depth domain knowledge. This study highlights the potential utility of LLMs for supporting software development, however, tasks requiring comprehensive, innovative or unconventional solutions, and careful debugging and error correction seem to be better developed by human programmers. We plot an agenda for the software engineering community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07739v3">SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 13 pages, 7 figures, 4 tables; Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Although many efforts have been made, it is still a challenge to balance the training budget, downstream performance, and the general capabilities of the LLMs in many applications. Training the whole model for downstream tasks is expensive, and could easily result in catastrophic forgetting. By introducing parameter-efficient fine-tuning (PEFT), the training cost could be reduced, but it still suffers from forgetting, and limits the learning on the downstream tasks. To efficiently fine-tune the LLMs with less limitation to their downstream performance while mitigating the forgetting of general capabilities, we propose a novel mixture of expert (MoE) framework based on Soft LoRA and Identity Mixture (SLIM), that allows dynamic routing between LoRA adapters and skipping connection, enables the suppression of forgetting. We adopt weight-yielding with sliding clustering for better out-of-domain distinguish to enhance the routing. We also propose to convert the mixture of low-rank adapters to the model merging formulation and introduce fast dynamic merging of LoRA adapters to keep the general capabilities of the base model. Extensive experiments demonstrate that the proposed SLIM is comparable to the state-of-the-art PEFT approaches on the downstream tasks while achieving the leading performance in mitigating catastrophic forgetting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17840v2">Is the GPU Half-Empty or Half-Full? Practical Scheduling Techniques for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 12 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Serving systems for Large Language Models (LLMs) improve throughput by processing several requests concurrently. However, multiplexing hardware resources between concurrent requests involves non-trivial scheduling decisions. Practical serving systems typically implement these decisions at two levels: First, a load balancer routes requests to different servers which each hold a replica of the LLM. Then, on each server, an engine-level scheduler decides when to run a request, or when to queue or preempt it. Improved scheduling policies may benefit a wide range of LLM deployments and can often be implemented as "drop-in replacements" to a system's current policy. In this work, we survey scheduling techniques from the literature and from practical serving systems. We find that schedulers from the literature often achieve good performance but introduce significant complexity. In contrast, schedulers in practical deployments often leave easy performance gains on the table but are easy to implement, deploy and configure. This finding motivates us to introduce two new scheduling techniques, which are both easy to implement, and outperform current techniques on production workload traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16783v1">A Stochastic Dynamical Theory of LLM Self-Adversariality: Modeling Severity Drift as a Critical Process</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 Experimental verification and more formal argument for Markov approximation of bias propagation to be released soon. Primarily pushed now to establish novelty and ease of sharing. Please do not cite this work until the forthcoming experimental validation and updated mathematical model are provided
    </div>
    <details class="paper-abstract">
      This paper introduces a continuous-time stochastic dynamical framework for understanding how large language models (LLMs) may self-amplify latent biases or toxicity through their own chain-of-thought reasoning. The model posits an instantaneous "severity" variable $x(t) \in [0,1]$ evolving under a stochastic differential equation (SDE) with a drift term $\mu(x)$ and diffusion $\sigma(x)$. Crucially, such a process can be consistently analyzed via the Fokker--Planck approach if each incremental step behaves nearly Markovian in severity space. The analysis investigates critical phenomena, showing that certain parameter regimes create phase transitions from subcritical (self-correcting) to supercritical (runaway severity). The paper derives stationary distributions, first-passage times to harmful thresholds, and scaling laws near critical points. Finally, it highlights implications for agents and extended LLM reasoning models: in principle, these equations might serve as a basis for formal verification of whether a model remains stable or propagates bias over repeated inferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16750v1">HateBench: Benchmarking Hate Speech Detectors on LLM-Generated Content and Hate Campaigns</a></div>
    <div class="paper-meta">
      📅 2025-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have raised increasing concerns about their misuse in generating hate speech. Among all the efforts to address this issue, hate speech detectors play a crucial role. However, the effectiveness of different detectors against LLM-generated hate speech remains largely unknown. In this paper, we propose HateBench, a framework for benchmarking hate speech detectors on LLM-generated hate speech. We first construct a hate speech dataset of 7,838 samples generated by six widely-used LLMs covering 34 identity groups, with meticulous annotations by three labelers. We then assess the effectiveness of eight representative hate speech detectors on the LLM-generated dataset. Our results show that while detectors are generally effective in identifying LLM-generated hate speech, their performance degrades with newer versions of LLMs. We also reveal the potential of LLM-driven hate campaigns, a new threat that LLMs bring to the field of hate speech detection. By leveraging advanced techniques like adversarial attacks and model stealing attacks, the adversary can intentionally evade the detector and automate hate campaigns online. The most potent adversarial attack achieves an attack success rate of 0.966, and its attack efficiency can be further improved by $13-21\times$ through model stealing attacks with acceptable attack performance. We hope our study can serve as a call to action for the research community and platform moderators to fortify defenses against these emerging threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16748v1">Through the Prism of Culture: Evaluating LLMs' Understanding of Indian Subcultures and Traditions</a></div>
    <div class="paper-meta">
      📅 2025-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements but also raise concerns about cultural bias, often reflecting dominant narratives at the expense of under-represented subcultures. In this study, we evaluate the capacity of LLMs to recognize and accurately respond to the Little Traditions within Indian society, encompassing localized cultural practices and subcultures such as caste, kinship, marriage, and religion. Through a series of case studies, we assess whether LLMs can balance the interplay between dominant Great Traditions and localized Little Traditions. We explore various prompting strategies and further investigate whether using prompts in regional languages enhances the models cultural sensitivity and response quality. Our findings reveal that while LLMs demonstrate an ability to articulate cultural nuances, they often struggle to apply this understanding in practical, context-specific scenarios. To the best of our knowledge, this is the first study to analyze LLMs engagement with Indian subcultures, offering critical insights into the challenges of embedding cultural diversity in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03168v3">Can Watermarked LLMs be Identified by Users via Crafted Prompts?</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 28 pages, 5 figures, 11 tables Published as a conference paper at ICLR 2025 Github link: https://github.com/THU-BPM/Watermarked_LLM_Identification
    </div>
    <details class="paper-abstract">
      Text watermarking for Large Language Models (LLMs) has made significant progress in detecting LLM outputs and preventing misuse. Current watermarking techniques offer high detectability, minimal impact on text quality, and robustness to text editing. However, current researches lack investigation into the imperceptibility of watermarking techniques in LLM services. This is crucial as LLM providers may not want to disclose the presence of watermarks in real-world scenarios, as it could reduce user willingness to use the service and make watermarks more vulnerable to attacks. This work is the first to investigate the imperceptibility of watermarked LLMs. We design an identification algorithm called Water-Probe that detects watermarks through well-designed prompts to the LLM. Our key motivation is that current watermarked LLMs expose consistent biases under the same watermark key, resulting in similar differences across prompts under different watermark keys. Experiments show that almost all mainstream watermarking algorithms are easily identified with our well-designed prompts, while Water-Probe demonstrates a minimal false positive rate for non-watermarked LLMs. Finally, we propose that the key to enhancing the imperceptibility of watermarked LLMs is to increase the randomness of watermark key selection. Based on this, we introduce the Water-Bag strategy, which significantly improves watermark imperceptibility by merging multiple watermark keys.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16744v1">LLM Assisted Anomaly Detection Service for Site Reliability Engineers: Enhancing Cloud Infrastructure Resilience</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 Accepted at the AAAI-2025 Deployable AI Workshop
    </div>
    <details class="paper-abstract">
      This paper introduces a scalable Anomaly Detection Service with a generalizable API tailored for industrial time-series data, designed to assist Site Reliability Engineers (SREs) in managing cloud infrastructure. The service enables efficient anomaly detection in complex data streams, supporting proactive identification and resolution of issues. Furthermore, it presents an innovative approach to anomaly modeling in cloud infrastructure by utilizing Large Language Models (LLMs) to understand key components, their failure modes, and behaviors. A suite of algorithms for detecting anomalies is offered in univariate and multivariate time series data, including regression-based, mixture-model-based, and semi-supervised approaches. We provide insights into the usage patterns of the service, with over 500 users and 200,000 API calls in a year. The service has been successfully applied in various industrial settings, including IoT-based AI applications. We have also evaluated our system on public anomaly benchmarks to show its effectiveness. By leveraging it, SREs can proactively identify potential issues before they escalate, reducing downtime and improving response times to incidents, ultimately enhancing the overall customer experience. We plan to extend the system to include time series foundation models, enabling zero-shot anomaly detection capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16698v1">3D-MoE: A Mixture-of-Experts Multi-modal LLM for 3D Vision and Pose Diffusion via Rectified Flow</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 Preprint. Work in progress
    </div>
    <details class="paper-abstract">
      3D vision and spatial reasoning have long been recognized as preferable for accurately perceiving our three-dimensional world, especially when compared with traditional visual reasoning based on 2D images. Due to the difficulties in collecting high-quality 3D data, research in this area has only recently gained momentum. With the advent of powerful large language models (LLMs), multi-modal LLMs for 3D vision have been developed over the past few years. However, most of these models focus primarily on the vision encoder for 3D data. In this paper, we propose converting existing densely activated LLMs into mixture-of-experts (MoE) models, which have proven effective for multi-modal data processing. In addition to leveraging these models' instruction-following capabilities, we further enable embodied task planning by attaching a diffusion head, Pose-DiT, that employs a novel rectified flow diffusion scheduler. Experimental results on 3D question answering and task-planning tasks demonstrate that our 3D-MoE framework achieves improved performance with fewer activated parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09003v3">Refusal in LLMs is an Affine Function</a></div>
    <div class="paper-meta">
      📅 2025-01-28
    </div>
    <details class="paper-abstract">
      We propose affine concept editing (ACE) as an approach for steering language models' behavior by intervening directly in activations. We begin with an affine decomposition of model activation vectors and show that prior methods for steering model behavior correspond to subsets of terms of this decomposition. We then provide a derivation of ACE and use it to control refusal behavior on ten different models, including Llama 3 70B. ACE combines affine subspace projection and activation addition to reliably control the model's refusal responses across prompt types. We evaluate the results using LLM-based scoring on a collection of harmful and harmless prompts. Our experiments demonstrate that ACE consistently achieves more precise control over model behavior than existing methods and generalizes to models where directional ablation via affine subspace projection alone produces incoherent outputs. Code for reproducing our results is available at https://github.com/EleutherAI/steering-llama3 .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16672v1">VeriFact: Verifying Facts in LLM-Generated Clinical Text with Electronic Health Records</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 62 pages, 5 figures, 1 table, pre-print manuscript
    </div>
    <details class="paper-abstract">
      Methods to ensure factual accuracy of text generated by large language models (LLM) in clinical medicine are lacking. VeriFact is an artificial intelligence system that combines retrieval-augmented generation and LLM-as-a-Judge to verify whether LLM-generated text is factually supported by a patient's medical history based on their electronic health record (EHR). To evaluate this system, we introduce VeriFact-BHC, a new dataset that decomposes Brief Hospital Course narratives from discharge summaries into a set of simple statements with clinician annotations for whether each statement is supported by the patient's EHR clinical notes. Whereas highest agreement between clinicians was 88.5%, VeriFact achieves up to 92.7% agreement when compared to a denoised and adjudicated average human clinican ground truth, suggesting that VeriFact exceeds the average clinician's ability to fact-check text against a patient's medical record. VeriFact may accelerate the development of LLM-based EHR applications by removing current evaluation bottlenecks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16661v1">Jupybara: Operationalizing a Design Space for Actionable Data Analysis and Storytelling with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 Conditionally Accepted at CHI '25
    </div>
    <details class="paper-abstract">
      Mining and conveying actionable insights from complex data is a key challenge of exploratory data analysis (EDA) and storytelling. To address this challenge, we present a design space for actionable EDA and storytelling. Synthesizing theory and expert interviews, we highlight how semantic precision, rhetorical persuasion, and pragmatic relevance underpin effective EDA and storytelling. We also show how this design space subsumes common challenges in actionable EDA and storytelling, such as identifying appropriate analytical strategies and leveraging relevant domain knowledge. Building on the potential of LLMs to generate coherent narratives with commonsense reasoning, we contribute Jupybara, an AI-enabled assistant for actionable EDA and storytelling implemented as a Jupyter Notebook extension. Jupybara employs two strategies -- design-space-aware prompting and multi-agent architectures -- to operationalize our design space. An expert evaluation confirms Jupybara's usability, steerability, explainability, and reparability, as well as the effectiveness of our strategies in operationalizing the design space framework with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16539v1">Generalized Mission Planning for Heterogeneous Multi-Robot Teams via LLM-constructed Hierarchical Trees</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      We present a novel mission-planning strategy for heterogeneous multi-robot teams, taking into account the specific constraints and capabilities of each robot. Our approach employs hierarchical trees to systematically break down complex missions into manageable sub-tasks. We develop specialized APIs and tools, which are utilized by Large Language Models (LLMs) to efficiently construct these hierarchical trees. Once the hierarchical tree is generated, it is further decomposed to create optimized schedules for each robot, ensuring adherence to their individual constraints and capabilities. We demonstrate the effectiveness of our framework through detailed examples covering a wide range of missions, showcasing its flexibility and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16534v1">Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we present and evaluate a method to assess the robustness of LLM alignment. We observe that alignment embeds a safety classifier in the target model that is responsible for deciding between refusal and compliance. We seek to extract an approximation of this classifier, called a surrogate classifier, from the LLM. We develop an algorithm for identifying candidate classifiers from subsets of the LLM model. We evaluate the degree to which the candidate classifiers approximate the model's embedded classifier in benign (F1 score) and adversarial (using surrogates in a white-box attack) settings. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find attacks mounted on the surrogate models can be transferred with high accuracy. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70%, a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is a viable (and highly effective) means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16533v1">A comparison of data filtering techniques for English-Polish LLM-based machine translation in the biomedical domain</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become state-of-the-art in Machine Translation (MT), often trained on massive bilingual parallel corpora scraped from the web, that contain low-quality entries and redundant information, leading to significant computational challenges. Various data filtering methods exist to reduce dataset sizes, but their effectiveness largely varies based on specific language pairs and domains. This paper evaluates the impact of commonly used data filtering techniques, such as LASER, MUSE, and LaBSE, on English-Polish translation within the biomedical domain. By filtering the UFAL Medical Corpus, we created varying dataset sizes to fine-tune the mBART50 model, which was then evaluated using the SacreBLEU metric on the Khresmoi dataset, having the quality of translations assessed by bilingual speakers. Our results show that both LASER and MUSE can significantly reduce dataset sizes while maintaining or even enhancing performance. We recommend the use of LASER, as it consistently outperforms the other methods and provides the most fluent and natural-sounding translations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16516v1">How well can LLMs Grade Essays in Arabic?</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      This research assesses the effectiveness of state-of-the-art large language models (LLMs), including ChatGPT, Llama, Aya, Jais, and ACEGPT, in the task of Arabic automated essay scoring (AES) using the AR-AES dataset. It explores various evaluation methodologies, including zero-shot, few-shot in-context learning, and fine-tuning, and examines the influence of instruction-following capabilities through the inclusion of marking guidelines within the prompts. A mixed-language prompting strategy, integrating English prompts with Arabic content, was implemented to improve model comprehension and performance. Among the models tested, ACEGPT demonstrated the strongest performance across the dataset, achieving a Quadratic Weighted Kappa (QWK) of 0.67, but was outperformed by a smaller BERT-based model with a QWK of 0.88. The study identifies challenges faced by LLMs in processing Arabic, including tokenization complexities and higher computational demands. Performance variation across different courses underscores the need for adaptive models capable of handling diverse assessment formats and highlights the positive impact of effective prompt engineering on improving LLM outputs. To the best of our knowledge, this study is the first to empirically evaluate the performance of multiple generative Large Language Models (LLMs) on Arabic essays using authentic student data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17200v1">Improving LLM Leaderboards with Psychometrical Methodology</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 53 pages, 10 figures, 6 tables
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) has necessitated the creation of benchmarks to evaluate their performance. These benchmarks resemble human tests and surveys, as they consist of sets of questions designed to measure emergent properties in the cognitive behavior of these systems. However, unlike the well-defined traits and abilities studied in social sciences, the properties measured by these benchmarks are often vaguer and less rigorously defined. The most prominent benchmarks are often grouped into leaderboards for convenience, aggregating performance metrics and enabling comparisons between models. Unfortunately, these leaderboards typically rely on simplistic aggregation methods, such as taking the average score across benchmarks. In this paper, we demonstrate the advantages of applying contemporary psychometric methodologies - originally developed for human tests and surveys - to improve the ranking of large language models on leaderboards. Using data from the Hugging Face Leaderboard as an example, we compare the results of the conventional naive ranking approach with a psychometrically informed ranking. The findings highlight the benefits of adopting psychometric techniques for more robust and meaningful evaluation of LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00215v3">HumanEvalComm: Benchmarking the Communication Competence of Code Generation for LLMs and LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 42 pages, 6 figures, 16 tables. Published in ACM Transactions on Software Engineering and Methodology (TOSEM)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly improved their ability to perform tasks in the field of code generation. However, there is still a gap between LLMs being capable coders and being top-tier software engineers. Based on the observation that top-level software engineers often ask clarifying questions to reduce ambiguity in both requirements and coding solutions, we argue that the same should be applied to LLMs for code generation tasks. In this work, we conducted an empirical study on the benchmark and analysis of the communication skills of LLMs for code generation. We define communication skills of LLMs as ``being able to ask clarifying questions when the description of the code generation problem has issues''. We created a new benchmark, HumanEvalComm, by modifying problem descriptions according to three issues: inconsistency, ambiguity, incompleteness. We defined new evaluation metrics such as Communication Rate and Good Question Rate, and then experimented on HumanEvalComm with different Code LLMs, and a new LLM agent approach, Okanagan, to identify and ask questions in ambiguous parts from code and descriptions for further refining the generated code. Finally, we discussed evaluation results by comparing Code LLMs and Okanagan with our findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.01990v4">From Prompt Injections to SQL Injection Attacks: How Protected is Your LLM-Integrated Web Application?</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 12 pages, 3 figures, 3 tables, 5 listings. 47th IEEE/ACM International Conference on Software Engineering (2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have found widespread applications in various domains, including web applications, where they facilitate human interaction via chatbots with natural language interfaces. Internally, aided by an LLM-integration middleware such as Langchain, user prompts are translated into SQL queries used by the LLM to provide meaningful responses to users. However, unsanitized user prompts can lead to SQL injection attacks, potentially compromising the security of the database. Despite the growing interest in prompt injection vulnerabilities targeting LLMs, the specific risks of generating SQL injection attacks through prompt injections have not been extensively studied. In this paper, we present a comprehensive examination of prompt-to-SQL (P$_2$SQL) injections targeting web applications based on the Langchain framework. Using Langchain as our case study, we characterize P$_2$SQL injections, exploring their variants and impact on application security through multiple concrete examples. Furthermore, we evaluate 7 state-of-the-art LLMs, demonstrating the pervasiveness of P$_2$SQL attacks across language models. Our findings indicate that LLM-integrated applications based on Langchain are highly susceptible to P$_2$SQL injection attacks, warranting the adoption of robust defenses. To counter these attacks, we propose four effective defense techniques that can be integrated as extensions to the Langchain framework. We validate the defenses through an experimental evaluation with a real-world use case application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03038v2">Towards Federated RLHF with Aggregated Client Preference for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 ICLR'25
    </div>
    <details class="paper-abstract">
      Reinforcement learning with human feedback (RLHF) fine-tunes a pretrained large language model (LLM) using user preference data, enabling it to generate content aligned with human preferences. However, due to privacy concerns, users may be reluctant to share sensitive preference data. To address this, we propose utilizing Federated Learning (FL) techniques, allowing large-scale preference collection from diverse real-world users without requiring them to transmit data to a central server. Our federated RLHF methods (i.e., FedBis and FedBiscuit) encode each client's preferences into binary selectors and aggregate them to capture common preferences. In particular, FedBiscuit overcomes key challenges, such as preference heterogeneity and reward hacking, through innovative solutions like grouping clients with similar preferences to reduce heterogeneity and using multiple binary selectors to enhance LLM output quality. To evaluate the performance of the proposed methods, we establish the first federated RLHF benchmark with a heterogeneous human preference dataset. Experimental results show that by integrating the LLM with aggregated client preferences, FedBis and FedBiscuit significantly enhance the professionalism and readability of the generated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16466v1">On the Feasibility of Using LLMs to Execute Multistage Network Attacks</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 16 pages, 14 figures
    </div>
    <details class="paper-abstract">
      LLMs have shown preliminary promise in some security tasks and CTF challenges. However, it is unclear whether LLMs are able to realize multistage network attacks, which involve executing a wide variety of actions across multiple hosts such as conducting reconnaissance, exploiting vulnerabilities to gain initial access, leveraging internal hosts to move laterally, and using multiple compromised hosts to exfiltrate data. We evaluate LLMs across 10 multistage networks and find that popular LLMs are unable to realize these attacks. To enable LLMs to realize these attacks, we introduce Incalmo, an LLM-agnostic high-level attack abstraction layer that sits between an LLM and the environment. Rather than LLMs issuing low-level command-line instructions, which can lead to incorrect implementations, Incalmo allows LLMs to specify high-level tasks (e.g., infect a host, scan a network), which are then carried out by Incalmo. Incalmo realizes these tasks by translating them into low-level primitives (e.g., commands to exploit tools). Incalmo also provides an environment state service and an attack graph service to provide structure to LLMs in selecting actions relevant to a multistage attack. Across 9 out of 10 realistic emulated networks (from 25 to 50 hosts), LLMs using Incalmo can successfully autonomously execute multistage attacks. We also conduct an ablation analysis to show the key role the high-level abstractions play. For instance, we find that both Incalmo's high-level tasks and services are crucial. Furthermore, even smaller-parameter LLMs with Incalmo can fully succeed in 5 of 10 environments, while larger-parameter LLMs without Incalmo do not fully succeed in any.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03471v2">MetRex: A Benchmark for Verilog Code Metric Reasoning Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been applied to various hardware design tasks, including Verilog code generation, EDA tool scripting, and RTL bug fixing. Despite this extensive exploration, LLMs are yet to be used for the task of post-synthesis metric reasoning and estimation of HDL designs. In this paper, we assess the ability of LLMs to reason about post-synthesis metrics of Verilog designs. We introduce MetRex, a large-scale dataset comprising 25,868 Verilog HDL designs and their corresponding post-synthesis metrics, namely area, delay, and static power. MetRex incorporates a Chain of Thought (CoT) template to enhance LLMs' reasoning about these metrics. Extensive experiments show that Supervised Fine-Tuning (SFT) boosts the LLM's reasoning capabilities on average by 37.0\%, 25.3\%, and 25.7\% on the area, delay, and static power, respectively. While SFT improves performance on our benchmark, it remains far from achieving optimal results, especially on complex problems. Comparing to state-of-the-art regression models, our approach delivers accurate post-synthesis predictions for 17.4\% more designs (within a 5\% error margin), in addition to offering a 1.7x speedup by eliminating the need for pre-processing. This work lays the groundwork for advancing LLM-based Verilog code metric reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16277v1">Do LLMs Have Visualization Literacy? An Evaluation on Modified Visualizations to Test Generalization in Data Interpretation</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 This is the author's version of the article that has been accepted in IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      In this paper, we assess the visualization literacy of two prominent Large Language Models (LLMs): OpenAI's Generative Pretrained Transformers (GPT), the backend of ChatGPT, and Google's Gemini, previously known as Bard, to establish benchmarks for assessing their visualization capabilities. While LLMs have shown promise in generating chart descriptions, captions, and design suggestions, their potential for evaluating visualizations remains under-explored. Collecting data from humans for evaluations has been a bottleneck for visualization research in terms of both time and money, and if LLMs were able to serve, even in some limited role, as evaluators, they could be a significant resource. To investigate the feasibility of using LLMs in the visualization evaluation process, we explore the extent to which LLMs possess visualization literacy -- a crucial factor for their effective utility in the field. We conducted a series of experiments using a modified 53-item Visualization Literacy Assessment Test (VLAT) for GPT-4 and Gemini. Our findings indicate that the LLMs we explored currently fail to achieve the same levels of visualization literacy when compared to data from the general public reported in VLAT, and LLMs heavily relied on their pre-existing knowledge to answer questions instead of utilizing the information provided by the visualization when answering questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16207v1">From Informal to Formal -- Incorporating and Evaluating LLMs on Natural Language Requirements to Verifiable Formal Proofs</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      The research in AI-based formal mathematical reasoning has shown an unstoppable growth trend. These studies have excelled in mathematical competitions like IMO, showing significant progress. However, these studies intertwined multiple skills simultaneously, i.e., problem-solving, reasoning, and writing formal specifications, making it hard to precisely identify the LLMs' strengths and weaknesses in each task. This paper focuses on formal verification, an immediate application scenario of formal reasoning, and decomposes it into six sub-tasks. We constructed 18k high-quality instruction-response pairs across five mainstream formal specification languages (Coq, Lean4, Dafny, ACSL, and TLA+) in six formal-verification-related tasks by distilling GPT-4o. They are split into a 14k+ fine-tuning dataset FM-alpaca and a 4k benchmark FM-Bench. We found that LLMs are good at writing proof segments when given either the code, or the detailed description of proof steps. Also, the fine-tuning brought about a nearly threefold improvement at most. Interestingly, we observed that fine-tuning with formal data also enhances mathematics, reasoning, and coding abilities. We hope our findings inspire further research. Fine-tuned models are released to facilitate subsequent studies
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19442v3">Does Generative AI speak Nigerian-Pidgin?: Issues about Representativeness and Bias for Multilingualism in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 Accepted to NAACL 2025 (findings)
    </div>
    <details class="paper-abstract">
      Nigeria is a multilingual country with 500+ languages. Naija is a Nigerian-Pidgin spoken by approx. 120M speakers in Nigeria and it is a mixed language (e.g., English, Portuguese, Yoruba, Hausa and Igbo). Although it has mainly been a spoken language until recently, there are now various platforms publishing exclusively in Naija such as Naija Wikipedia. However, it is hard to distinguish by non-native from a larger pidgin languages spoken across West Africa known as West African Pidgin English (WAPE) -- which is more simplied and understandable by wider audience in Ghana, Nigeria, and Cameroon. BBC news platform publishes exclusively in WAPE to cater for several countries in West Africa. In our paper, we show through statistical analyses and Machine Translation experiments that these two creole varieties do not represent each other (i.e., there are linguistic differences in word order and vocabulary) and Generative AI operates only based on WAPE. In other words, Naija is under-represented in Generative AI, and it is hard to teach LLMs with few examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16191v1">Raiders of the Lost Dependency: Fixing Dependency Conflicts in Python using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 Under submission to TOSEM, 2025
    </div>
    <details class="paper-abstract">
      Fixing Python dependency issues is a tedious and error-prone task for developers, who must manually identify and resolve environment dependencies and version constraints of third-party modules and Python interpreters. Researchers have attempted to automate this process by relying on large knowledge graphs and database lookup tables. However, these traditional approaches face limitations due to the variety of dependency error types, large sets of possible module versions, and conflicts among transitive dependencies. This study explores the potential of using large language models (LLMs) to automatically fix dependency issues in Python programs. We introduce PLLM (pronounced "plum"), a novel technique that employs retrieval-augmented generation (RAG) to help an LLM infer Python versions and required modules for a given Python file. PLLM builds a testing environment that iteratively (1) prompts the LLM for module combinations, (2) tests the suggested changes, and (3) provides feedback (error messages) to the LLM to refine the fix. This feedback cycle leverages natural language processing (NLP) to intelligently parse and interpret build error messages. We benchmark PLLM on the Gistable HG2.9K dataset, a collection of challenging single-file Python gists. We compare PLLM against two state-of-the-art automatic dependency inference approaches, namely PyEGo and ReadPyE, w.r.t. the ability to resolve dependency issues. Our results indicate that PLLM can fix more dependency issues than the two baselines, with +218 (+15.97%) more fixes over ReadPyE and +281 (+21.58%) over PyEGo. Our deeper analyses suggest that PLLM is particularly beneficial for projects with many dependencies and for specific third-party numerical and machine-learning modules. Our findings demonstrate the potential of LLM-based approaches to iteratively resolve Python dependency issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10259v4">Uncovering Latent Arguments in Social Media Messaging by Employing LLMs-in-the-Loop Strategy</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 Accepted at the Findings of 2025 Annual Conference of the Nations of the Americas Chapter of the ACL (NAACL 2025)
    </div>
    <details class="paper-abstract">
      The widespread use of social media has led to a surge in popularity for automated methods of analyzing public opinion. Supervised methods are adept at text categorization, yet the dynamic nature of social media discussions poses a continual challenge for these techniques due to the constant shifting of the focus. On the other hand, traditional unsupervised methods for extracting themes from public discourse, such as topic modeling, often reveal overarching patterns that might not capture specific nuances. Consequently, a significant portion of research into social media discourse still depends on labor-intensive manual coding techniques and a human-in-the-loop approach, which are both time-consuming and costly. In this work, we study the problem of discovering arguments associated with a specific theme. We propose a generic LLMs-in-the-Loop strategy that leverages the advanced capabilities of Large Language Models (LLMs) to extract latent arguments from social media messaging. To demonstrate our approach, we apply our framework to contentious topics. We use two publicly available datasets: (1) the climate campaigns dataset of 14k Facebook ads with 25 themes and (2) the COVID-19 vaccine campaigns dataset of 9k Facebook ads with 14 themes. Additionally, we design a downstream task as stance prediction by leveraging talking points in climate debates. Furthermore, we analyze demographic targeting and the adaptation of messaging based on real-world events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16173v1">Will Systems of LLM Agents Cooperate: An Investigation into a Social Dilemma</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 7 pages (10 including references), 4 figures
    </div>
    <details class="paper-abstract">
      As autonomous agents become more prevalent, understanding their collective behaviour in strategic interactions is crucial. This study investigates the emergent cooperative tendencies of systems of Large Language Model (LLM) agents in a social dilemma. Unlike previous research where LLMs output individual actions, we prompt state-of-the-art LLMs to generate complete strategies for iterated Prisoner's Dilemma. Using evolutionary game theory, we simulate populations of agents with different strategic dispositions (aggressive, cooperative, or neutral) and observe their evolutionary dynamics. Our findings reveal that different LLMs exhibit distinct biases affecting the relative success of aggressive versus cooperative strategies. This research provides insights into the potential long-term behaviour of systems of deployed LLM-based autonomous agents and highlights the importance of carefully considering the strategic environments in which they operate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16155v1">CITYWALK: Enhancing LLM-Based C++ Unit Test Generation via Project-Dependency Awareness and Language-Specific Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 13 tables, 12 figures. Under Review
    </div>
    <details class="paper-abstract">
      Unit testing plays a pivotal role in the software development lifecycle, as it ensures code quality. However, writing high-quality unit tests remains a time-consuming task for developers in practice. More recently, the application of large language models (LLMs) in automated unit test generation has demonstrated promising results. Existing approaches primarily focus on interpreted programming languages (e.g., Java), while mature solutions tailored to compiled programming languages like C++ are yet to be explored. The intricate language features of C++, such as pointers, templates, and virtual functions, pose particular challenges for LLMs in generating both executable and high-coverage unit tests. To tackle the aforementioned problems, this paper introduces CITYWALK, a novel LLM-based framework for C++ unit test generation. CITYWALK enhances LLMs by providing a comprehensive understanding of the dependency relationships within the project under test via program analysis. Furthermore, CITYWALK incorporates language-specific knowledge about C++ derived from project documentation and empirical observations, significantly improving the correctness of the LLM-generated unit tests. We implement CITYWALK by employing the widely popular LLM GPT-4o. The experimental results show that CITYWALK outperforms current state-of-the-art approaches on a collection of eight popular C++ projects. Our findings demonstrate the effectiveness of CITYWALK in generating high-quality C++ unit tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16078v1">Integration of LLM Quality Assurance into an NLG System</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      In this paper, we present a system that uses a Large Language Model (LLM) to perform grammar and spelling correction as a component of Quality Assurance (QA) for texts generated by NLG systems, which is important for text production in real-world scenarios. Evaluating the results of the system on work-in-progress sports news texts in three languages, we show that it is able to deliver acceptable corrections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16033v1">PRISMe: A Novel LLM-Powered Tool for Interactive Privacy Policy Assessment</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Protecting online privacy requires users to engage with and comprehend website privacy policies, but many policies are difficult and tedious to read. We present PRISMe (Privacy Risk Information Scanner for Me), a novel Large Language Model (LLM)-driven privacy policy assessment tool, which helps users to understand the essence of a lengthy, complex privacy policy while browsing. The tool, a browser extension, integrates a dashboard and an LLM chat. One major contribution is the first rigorous evaluation of such a tool. In a mixed-methods user study (N=22), we evaluate PRISMe's efficiency, usability, understandability of the provided information, and impacts on awareness. While our tool improves privacy awareness by providing a comprehensible quick overview and a quality chat for in-depth discussion, users note issues with consistency and building trust in the tool. From our insights, we derive important design implications to guide future policy analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16029v1">FDLLM: A Text Fingerprint Detection Method for LLMs in Multi-Language, Multi-Domain Black-Box Environments</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      Using large language models (LLMs) integration platforms without transparency about which LLM is being invoked can lead to potential security risks. Specifically, attackers may exploit this black-box scenario to deploy malicious models and embed viruses in the code provided to users. In this context, it is increasingly urgent for users to clearly identify the LLM they are interacting with, in order to avoid unknowingly becoming victims of malicious models. However, existing studies primarily focus on mixed classification of human and machine-generated text, with limited attention to classifying texts generated solely by different models. Current research also faces dual bottlenecks: poor quality of LLM-generated text (LLMGT) datasets and limited coverage of detectable LLMs, resulting in poor detection performance for various LLMGT in black-box scenarios. We propose the first LLMGT fingerprint detection model, \textbf{FDLLM}, based on Qwen2.5-7B and fine-tuned using LoRA to address these challenges. FDLLM can more efficiently handle detection tasks across multilingual and multi-domain scenarios. Furthermore, we constructed a dataset named \textbf{FD-Datasets}, consisting of 90,000 samples that span multiple languages and domains, covering 20 different LLMs. Experimental results demonstrate that FDLLM achieves a macro F1 score 16.7\% higher than the best baseline method, LM-D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15044v2">Adanonymizer: Interactively Navigating and Balancing the Duality of Privacy and Output Performance in Human-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) cannot support users to precisely balance privacy protection and output performance during individual consultations. We introduce Adanonymizer, an anonymization plug-in that allows users to control this balance by navigating a trade-off curve. A survey (N=221) revealed a privacy paradox, where users frequently disclosed sensitive information despite acknowledging privacy risks. The study further demonstrated that privacy risks were not significantly correlated with model output performance, highlighting the potential to navigate this trade-off. Adanonymizer normalizes privacy and utility ratings by type and automates the pseudonymization of sensitive terms based on user preferences, significantly reducing user effort. Its 2D color palette interface visualizes the privacy-utility trade-off, allowing users to adjust the balance by manipulating a point. An evaluation (N=36) compared Adanonymizer with ablation methods and differential privacy techniques, where Adanonymizer significantly reduced modification time, achieved better perceived model performance and overall user preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02337v3">WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable potential as autonomous agents, particularly in web-based tasks. However, existing LLM web agents heavily rely on expensive proprietary LLM APIs, while open LLMs lack the necessary decision-making capabilities. This paper introduces WebRL, a self-evolving online curriculum reinforcement learning framework designed to train high-performance web agents using open LLMs. WebRL addresses three key challenges in building LLM web agents, including the scarcity of training tasks, sparse feedback signals, and policy distribution drift in online learning. Specifically, WebRL incorporates 1) a self-evolving curriculum that generates new tasks from unsuccessful attempts, 2) a robust outcome-supervised reward model (ORM), and 3) adaptive reinforcement learning strategies to ensure consistent improvements. We apply WebRL to transform open Llama-3.1 and GLM-4 models into proficient web agents. On WebArena-Lite, WebRL improves the success rate of Llama-3.1-8B from 4.8% to 42.4%, and from 6.1% to 43% for GLM-4-9B. These open models significantly surpass the performance of GPT-4-Turbo (17.6%) and GPT-4o (13.9%) and outperform previous state-of-the-art web agents trained on open LLMs (AutoWebGLM, 18.2%). Our findings demonstrate WebRL's effectiveness in bridging the gap between open and proprietary LLM-based web agents, paving the way for more accessible and powerful autonomous web interaction systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15268v3">Style Outweighs Substance: Failure Modes of LLM Judges in Alignment Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      The release of ChatGPT in November 2022 sparked an explosion of interest in post-training and an avalanche of new preference optimization (PO) methods. These methods claim superior alignment by virtue of better correspondence with human pairwise preferences, often measured by LLM-judges. In this work, we attempt to answer the following question -- do LLM-judge preferences translate to progress on other, more concrete metrics for alignment, and if not, why not? We define a concrete metric for alignment, and introduce SOS-Bench (Substance Outweighs Style Benchmark), which is to the best of our knowledge the largest standardized, reproducible LLM meta-benchmark to date. We find that (1) LLM-judge preferences do not correlate with concrete measures of safety, world knowledge, and instruction following; (2) LLM-judges have powerful implicit biases, prioritizing style over factuality and safety; and (3) the supervised fine-tuning (SFT) stage of post-training, and not the PO stage, has the greatest impact on alignment, with data scaling and prompt diversity as the driving factors. Our codebase and complete results can be found at https://github.com/penfever/sos-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12030v3">Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) of several billion parameters can be impractical in most industrial use cases due to constraints such as cost, latency limitations, and hardware accessibility. Knowledge distillation (KD) offers a solution by compressing knowledge from resource-intensive large models to smaller ones. Various strategies exist, some relying on the text generated by the teacher model and optionally utilizing his logits to enhance learning. However, these methods based on logits often require both teacher and student models to share the same tokenizer, limiting their applicability across different LLM families. In this paper, we introduce Universal Logit Distillation (ULD) loss, grounded in optimal transport, to address this limitation. Our experimental results demonstrate the effectiveness of ULD loss in enabling distillation across models with different architectures and tokenizers, paving the way to a more widespread use of distillation techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15901v1">Robust Mobile Robot Path Planning via LLM-Based Dynamic Waypoint Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 18 pages, 6 figures, submitted in Journal Expert Systems with Applications
    </div>
    <details class="paper-abstract">
      Mobile robot path planning in complex environments remains a significant challenge, especially in achieving efficient, safe and robust paths. The traditional path planning techniques like DRL models typically trained for a given configuration of the starting point and target positions, these models only perform well when these conditions are satisfied. In this paper, we proposed a novel path planning framework that embeds Large Language Models to empower mobile robots with the capability of dynamically interpreting natural language commands and autonomously generating efficient, collision-free navigation paths. The proposed framework uses LLMs to translate high-level user inputs into actionable waypoints while dynamically adjusting paths in response to obstacles. We experimentally evaluated our proposed LLM-based approach across three different environments of progressive complexity, showing the robustness of our approach with llama3.1 model that outperformed other LLM models in path planning time, waypoint generation success rate, and collision avoidance. This underlines the promising contribution of LLMs for enhancing the capability of mobile robots, especially when their operation involves complex decisions in large and complex environments. Our framework has provided safer, more reliable navigation systems and opened a new direction for the future research. The source code of this work is publicly available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15875v1">LCTG Bench: LLM Controlled Text Generation Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 15 pages, 11 figures. Project page: this [URL](https://github.com/CyberAgentAILab/LCTG-Bench)
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has led to more diverse and higher-quality machine-generated text. However, their high expressive power makes it difficult to control outputs based on specific business instructions. In response, benchmarks focusing on the controllability of LLMs have been developed, but several issues remain: (1) They primarily cover major languages like English and Chinese, neglecting low-resource languages like Japanese; (2) Current benchmarks employ task-specific evaluation metrics, lacking a unified framework for selecting models based on controllability across different use cases. To address these challenges, this research introduces LCTG Bench, the first Japanese benchmark for evaluating the controllability of LLMs. LCTG Bench provides a unified framework for assessing control performance, enabling users to select the most suitable model for their use cases based on controllability. By evaluating nine diverse Japanese-specific and multilingual LLMs like GPT-4, we highlight the current state and challenges of controllability in Japanese LLMs and reveal the significant gap between multilingual models and Japanese-specific models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15829v1">Aging-aware CPU Core Management for Embodied Carbon Amortization in Cloud LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      Broad adoption of Large Language Models (LLM) demands rapid expansions of cloud LLM inference clusters, leading to accumulation of embodied carbon$-$the emissions from manufacturing and supplying IT assets$-$that mostly concentrate on inference server CPU. This paper delves into the challenges of sustainable growth of cloud LLM inference, emphasizing extended amortization of CPU embodied over an increased lifespan. Given the reliability risks of silicon aging, we propose an aging-aware CPU core management technique to delay CPU aging effects, allowing the cluster operator to safely increase CPU life. Our technique exploits CPU underutilization patterns that we uncover in cloud LLM inference by halting aging in unused cores and even-outing aging in active cores via selective deep idling and aging-aware inference task allocation. Through extensive simulations using real-world Azure inference traces and an extended LLM cluster simulator from Microsoft, we show superior performance of our technique over existing methods with an estimated 37.67\% reduction in yearly embodied carbon emissions through p99 performance of managing CPU aging effects, a 77\% reduction in CPU underutilization, and less than 10\% impact to the inference service quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03775v3">Beyond correlation: The Impact of Human Uncertainty in Measuring the Effectiveness of Automatic Evaluation and LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 Accepted at ICLR 2025
    </div>
    <details class="paper-abstract">
      The effectiveness of automatic evaluation of generative models is typically measured by comparing the labels generated via automation with labels by humans using correlation metrics. However, metrics like Krippendorff's $\alpha$ and Randolph's $\kappa$ were originally designed to measure the reliability of human labeling, thus make assumptions about typical human labeling behavior, and these assumptions may not be applicable to machine generated labels. In this paper, we show how *relying on a single aggregate correlation score* can obscure fundamental differences between human labels and those from automatic evaluation, including LLM-as-a-Judge. Specifically, we demonstrate that when the proportion of samples with variation or uncertainty in human assigned labels is relatively high, machine labels (generated by automatic evaluation methods) may superficially appear to have similar or better correlation with the human majority label compared to the human-to-human (HH) correlation. This can create the illusion that labels from automatic evaluation approximates the human majority label. However, as the proportion of samples with consistent human labels increases, the correlation between machine and human labels fall well below HH correlation. Based on these findings, we first propose stratifying data by human label uncertainty to provide a more robust analysis of automatic evaluation performance. Second, recognizing that uncertainty and variation are inherent in perception-based human evaluations, such as those involving attitudes or preferences, we introduce a new metric - binned Jensen-Shannon Divergence for perception for such scenarios to better measure the effectiveness of automatic evaluations. We present visualization techniques -- perception charts, to contextualize correlation measures appropriately. We have open-sourced at https://github.com/amazon-science/BeyondCorrelation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08705v4">When LLM Meets DRL: Advancing Jailbreaking Efficiency via DRL-guided Search</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      Recent studies developed jailbreaking attacks, which construct jailbreaking prompts to fool LLMs into responding to harmful questions. Early-stage jailbreaking attacks require access to model internals or significant human efforts. More advanced attacks utilize genetic algorithms for automatic and black-box attacks. However, the random nature of genetic algorithms significantly limits the effectiveness of these attacks. In this paper, we propose RLbreaker, a black-box jailbreaking attack driven by deep reinforcement learning (DRL). We model jailbreaking as a search problem and design an RL agent to guide the search, which is more effective and has less randomness than stochastic search, such as genetic algorithms. Specifically, we design a customized DRL system for the jailbreaking problem, including a novel reward function and a customized proximal policy optimization (PPO) algorithm. Through extensive experiments, we demonstrate that RLbreaker is much more effective than existing jailbreaking attacks against six state-of-the-art (SOTA) LLMs. We also show that RLbreaker is robust against three SOTA defenses and its trained agents can transfer across different LLMs. We further validate the key design choices of RLbreaker via a comprehensive ablation study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13879v2">Crabs: Consuming Resrouce via Auto-generation for LLM-DoS Attack under Black-box Settings</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 20 pages, 7 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks. LLMs continue to be vulnerable to external threats, particularly Denial-of-Service (DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, prior works tend to focus on performing white-box attacks, overlooking black-box settings. In this work, we propose an automated algorithm designed for black-box LLMs, called Auto-Generation for LLM-DoS Attack (AutoDoS). AutoDoS introduces DoS Attack Tree and optimizes the prompt node coverage to enhance effectiveness under black-box conditions. Our method can bypass existing defense with enhanced stealthiness via semantic improvement of prompt nodes. Furthermore, we reveal that implanting Length Trojan in Basic DoS Prompt aids in achieving higher attack efficacy. Experimental results show that AutoDoS amplifies service response latency by over 250 $\times \uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our code is available at https://github.com/shuita2333/AutoDoS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15755v1">GraphICL: Unlocking Graph Learning Potential in LLMs through Structured Prompt Design</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      The growing importance of textual and relational systems has driven interest in enhancing large language models (LLMs) for graph-structured data, particularly Text-Attributed Graphs (TAGs), where samples are represented by textual descriptions interconnected by edges. While research has largely focused on developing specialized graph LLMs through task-specific instruction tuning, a comprehensive benchmark for evaluating LLMs solely through prompt design remains surprisingly absent. Without such a carefully crafted evaluation benchmark, most if not all, tailored graph LLMs are compared against general LLMs using simplistic queries (e.g., zero-shot reasoning with LLaMA), which can potentially camouflage many advantages as well as unexpected predicaments of them. To achieve more general evaluations and unveil the true potential of LLMs for graph tasks, we introduce Graph In-context Learning (GraphICL) Benchmark, a comprehensive benchmark comprising novel prompt templates designed to capture graph structure and handle limited label knowledge. Our systematic evaluation shows that general-purpose LLMs equipped with our GraphICL outperform state-of-the-art specialized graph LLMs and graph neural network models in resource-constrained settings and out-of-domain tasks. These findings highlight the significant potential of prompt engineering to enhance LLM performance on graph learning tasks without training and offer a strong baseline for advancing research in graph LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15749v1">LLM-powered Multi-agent Framework for Goal-oriented Learning in Intelligent Tutoring System</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 Accepted by WWW 2025 (Industry Track)
    </div>
    <details class="paper-abstract">
      Intelligent Tutoring Systems (ITSs) have revolutionized education by offering personalized learning experiences. However, as goal-oriented learning, which emphasizes efficiently achieving specific objectives, becomes increasingly important in professional contexts, existing ITSs often struggle to deliver this type of targeted learning experience. In this paper, we propose GenMentor, an LLM-powered multi-agent framework designed to deliver goal-oriented, personalized learning within ITS. GenMentor begins by accurately mapping learners' goals to required skills using a fine-tuned LLM trained on a custom goal-to-skill dataset. After identifying the skill gap, it schedules an efficient learning path using an evolving optimization approach, driven by a comprehensive and dynamic profile of learners' multifaceted status. Additionally, GenMentor tailors learning content with an exploration-drafting-integration mechanism to align with individual learner needs. Extensive automated and human evaluations demonstrate GenMentor's effectiveness in learning guidance and content quality. Furthermore, we have deployed it in practice and also implemented it as an application. Practical human study with professional learners further highlights its effectiveness in goal alignment and resource targeting, leading to enhanced personalization. Supplementary resources are available at https://github.com/GeminiLight/gen-mentor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15674v1">TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      The reasoning abilities of Large Language Models (LLMs) can be improved by structurally denoising their weights, yet existing techniques primarily focus on denoising the feed-forward network (FFN) of the transformer block, and can not efficiently utilise the Multi-head Attention (MHA) block, which is the core of transformer architectures. To address this issue, we propose a novel intuitive framework that, at its very core, performs MHA compression through a multi-head tensorisation process and the Tucker decomposition. This enables both higher-dimensional structured denoising and compression of the MHA weights, by enforcing a shared higher-dimensional subspace across the weights of the multiple attention heads. We demonstrate that this approach consistently enhances the reasoning capabilities of LLMs across multiple benchmark datasets, and for both encoder-only and decoder-only architectures, while achieving compression rates of up to $\sim 250$ times in the MHA weights, all without requiring any additional data, training, or fine-tuning. Furthermore, we show that the proposed method can be seamlessly combined with existing FFN-only-based denoising techniques to achieve further improvements in LLM reasoning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17187v1">Visualizing Uncertainty in Translation Tasks: An Evaluation of LLM Performance and Confidence Metrics</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly utilized for machine translation, yet their predictions often exhibit uncertainties that hinder interpretability and user trust. Effectively visualizing these uncertainties can enhance the usability of LLM outputs, particularly in contexts where translation accuracy is critical. This paper addresses two primary objectives: (1) providing users with token-level insights into model confidence and (2) developing a web-based visualization tool to quantify and represent translation uncertainties. To achieve these goals, we utilized the T5 model with the WMT19 dataset for translation tasks and evaluated translation quality using established metrics such as BLEU, METEOR, and ROUGE. We introduced three novel uncertainty quantification (UQ) metrics: (1) the geometric mean of token probabilities, (2) the arithmetic mean of token probabilities, and (3) the arithmetic mean of the kurtosis of token distributions. These metrics provide a simple yet effective framework for evaluating translation performance. Our analysis revealed a linear relationship between the traditional evaluation metrics and our UQ metrics, demonstrating the validity of our approach. Additionally, we developed an interactive web-based visualization that uses a color gradient to represent token confidence. This tool offers users a clear and intuitive understanding of translation quality while providing valuable insights into model performance. Overall, we show that our UQ metrics and visualization are both robust and interpretable, offering practical tools for evaluating and accessing machine translation systems.
    </details>
</div>
