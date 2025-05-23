# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17005v1">R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful but prone to hallucinations due to static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting external information, but current methods often are costly, generalize poorly, or ignore the internal knowledge of the model. In this paper, we introduce R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage both internal and external knowledge sources. R1-Searcher++ employs a two-stage training strategy: an initial SFT Cold-start phase for preliminary format learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses outcome-supervision to encourage exploration, incorporates a reward mechanism for internal knowledge utilization, and integrates a memorization mechanism to continuously assimilate retrieved information, thereby enriching the model's internal knowledge. By leveraging internal knowledge and external search engine, the model continuously improves its capabilities, enabling efficient retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++ outperforms previous RAG and reasoning methods and achieves efficient retrieval. The code is available at https://github.com/RUCAIBox/R1-Searcher-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16997v1">X-MAS: Towards Building Multi-Agent Systems with Heterogeneous LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 19 pages, 5 figures
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) extend the capabilities of single LLMs by enabling cooperation among multiple specialized agents. However, most existing MAS frameworks rely on a single LLM to drive all agents, constraining the system's intelligence to the limit of that model. This paper explores the paradigm of heterogeneous LLM-driven MAS (X-MAS), where agents are powered by diverse LLMs, elevating the system's potential to the collective intelligence of diverse LLMs. We introduce X-MAS-Bench, a comprehensive testbed designed to evaluate the performance of various LLMs across different domains and MAS-related functions. As an extensive empirical study, we assess 27 LLMs across 5 domains (encompassing 21 test sets) and 5 functions, conducting over 1.7 million evaluations to identify optimal model selections for each domain-function combination. Building on these findings, we demonstrate that transitioning from homogeneous to heterogeneous LLM-driven MAS can significantly enhance system performance without requiring structural redesign. Specifically, in a chatbot-only MAS scenario, the heterogeneous configuration yields up to 8.4\% performance improvement on the MATH dataset. In a mixed chatbot-reasoner scenario, the heterogeneous MAS could achieve a remarkable 47\% performance boost on the AIME dataset. Our results underscore the transformative potential of heterogeneous LLMs in MAS, highlighting a promising avenue for advancing scalable, collaborative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16988v1">MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 18 pages, 11 figures
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) have demonstrated significant potential in enhancing single LLMs to address complex and diverse tasks in practical applications. Despite considerable advancements, the field lacks a unified codebase that consolidates existing methods, resulting in redundant re-implementation efforts, unfair comparisons, and high entry barriers for researchers. To address these challenges, we introduce MASLab, a unified, comprehensive, and research-friendly codebase for LLM-based MAS. (1) MASLab integrates over 20 established methods across multiple domains, each rigorously validated by comparing step-by-step outputs with its official implementation. (2) MASLab provides a unified environment with various benchmarks for fair comparisons among methods, ensuring consistent inputs and standardized evaluation protocols. (3) MASLab implements methods within a shared streamlined structure, lowering the barriers for understanding and extension. Building on MASLab, we conduct extensive experiments covering 10+ benchmarks and 8 models, offering researchers a clear and comprehensive view of the current landscape of MAS methods. MASLab will continue to evolve, tracking the latest developments in the field, and invite contributions from the broader open-source community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16983v1">LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are primarily designed for batch processing. Existing methods for adapting LLMs to streaming rely either on expensive re-encoding or specialized architectures with limited scalability. This work identifies three key mismatches in adapting batch-oriented LLMs to streaming: (1) input-attention, (2) output-attention, and (3) position-ID mismatches. While it is commonly assumed that the latter two mismatches require frequent re-encoding, our analysis reveals that only the input-attention mismatch significantly impacts performance, indicating re-encoding outputs is largely unnecessary. To better understand this discrepancy with the common assumption, we provide the first comprehensive analysis of the impact of position encoding on LLMs in streaming, showing that preserving relative positions within source and target contexts is more critical than maintaining absolute order. Motivated by the above analysis, we introduce a group position encoding paradigm built on batch architectures to enhance consistency between streaming and batch modes. Extensive experiments on cross-lingual and cross-modal tasks demonstrate that our method outperforms existing approaches. Our method requires no architectural modifications, exhibits strong generalization in both streaming and batch modes. The code is available at repository https://github.com/EIT-NLP/StreamingLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16979v1">Know the Ropes: A Heuristic Strategy for LLM-based Multi-Agent System Design</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Single-agent LLMs hit hard limits--finite context, role overload, and brittle domain transfer. Conventional multi-agent fixes soften those edges yet expose fresh pains: ill-posed decompositions, fuzzy contracts, and verification overhead that blunts the gains. We therefore present Know-The-Ropes (KtR), a framework that converts domain priors into an algorithmic blueprint hierarchy, in which tasks are recursively split into typed, controller-mediated subtasks, each solved zero-shot or with the lightest viable boost (e.g., chain-of-thought, micro-tune, self-check). Grounded in the No-Free-Lunch theorem, KtR trades the chase for a universal prompt for disciplined decomposition. On the Knapsack problem (3-8 items), three GPT-4o-mini agents raise accuracy from 3% zero-shot to 95% on size-5 instances after patching a single bottleneck agent. On the tougher Task-Assignment problem (6-15 jobs), a six-agent o3-mini blueprint hits 100% up to size 10 and 84% on sizes 13-15, versus 11% zero-shot. Algorithm-aware decomposition plus targeted augmentation thus turns modest models into reliable collaborators--no ever-larger monoliths required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16978v1">HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted to ACL 2025 Findings. Code available at https://github.com/RutaTang/HyGenar
    </div>
    <details class="paper-abstract">
      Grammar plays a critical role in natural language processing and text/code generation by enabling the definition of syntax, the creation of parsers, and guiding structured outputs. Although large language models (LLMs) demonstrate impressive capabilities across domains, their ability to infer and generate grammars has not yet been thoroughly explored. In this paper, we aim to study and improve the ability of LLMs for few-shot grammar generation, where grammars are inferred from sets of a small number of positive and negative examples and generated in Backus-Naur Form. To explore this, we introduced a novel dataset comprising 540 structured grammar generation challenges, devised 6 metrics, and evaluated 8 various LLMs against it. Our findings reveal that existing LLMs perform sub-optimally in grammar generation. To address this, we propose an LLM-driven hybrid genetic algorithm, namely HyGenar, to optimize grammar generation. HyGenar achieves substantial improvements in both the syntactic and semantic correctness of generated grammars across LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14625v2">TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become a powerful tool for enhancing the reasoning abilities of large language models (LLMs) by optimizing their policies with reward signals. Yet, RL's success relies on the reliability of rewards, which are provided by verifiers. In this paper, we expose and analyze a widespread problem--false negatives--where verifiers wrongly reject correct model outputs. Our in-depth study of the Big-Math-RL-Verified dataset reveals that over 38% of model-generated responses suffer from false negatives, where the verifier fails to recognize correct answers. We show, both empirically and theoretically, that these false negatives severely impair RL training by depriving the model of informative gradient signals and slowing convergence. To mitigate this, we propose tinyV, a lightweight LLM-based verifier that augments existing rule-based methods, which dynamically identifies potential false negatives and recovers valid responses to produce more accurate reward estimates. Across multiple math-reasoning benchmarks, integrating TinyV boosts pass rates by up to 10% and accelerates convergence relative to the baseline. Our findings highlight the critical importance of addressing verifier false negatives and offer a practical approach to improve RL-based fine-tuning of LLMs. Our code is available at https://github.com/uw-nsl/TinyV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16967v1">Fixing Data That Hurts Performance: Cascading LLMs to Relabel Hard Negatives for Robust Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Code is available at https://github.com/castorini/rlhn & datasets are available at https://huggingface.co/rlhn
    </div>
    <details class="paper-abstract">
      Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection reduces the training set size by 2.35$\times$ and increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We propose a simple, cost-effective approach using cascading LLM prompts to identify and relabel hard negatives. Experimental results show that relabeling false negatives with true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7-1.4 nDCG@10 on BEIR and by 1.7-1.8 nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of the cascading design is further supported by human annotation results, where we find judgment by GPT-4o shows much higher agreement with humans than GPT-4o-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16954v1">Cracking Aegis: An Adversarial LLM-based Game for Raising Awareness of Vulnerabilities in Privacy Protection</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 24 pages, In Designing Interactive Systems Conference (DIS 25)
    </div>
    <details class="paper-abstract">
      Traditional methods for raising awareness of privacy protection often fail to engage users or provide hands-on insights into how privacy vulnerabilities are exploited. To address this, we incorporate an adversarial mechanic in the design of the dialogue-based serious game Cracking Aegis. Leveraging LLMs to simulate natural interactions, the game challenges players to impersonate characters and extract sensitive information from an AI agent, Aegis. A user study (n=22) revealed that players employed diverse deceptive linguistic strategies, including storytelling and emotional rapport, to manipulate Aegis. After playing, players reported connecting in-game scenarios with real-world privacy vulnerabilities, such as phishing and impersonation, and expressed intentions to strengthen privacy control, such as avoiding oversharing personal information with AI systems. This work highlights the potential of LLMs to simulate complex relational interactions in serious games, while demonstrating how an adversarial game strategy provides unique insights for designs for social good, particularly privacy protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16947v1">MixAT: Combining Continuous and Discrete Adversarial Training for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14652v3">General-Reasoner: Advancing LLM Reasoning Across All Domains</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16894v1">Shadows in the Attention: Contextual Perturbation and Representation Drift in the Dynamics of Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Hallucinations -- plausible yet erroneous outputs -- remain a critical barrier to reliable deployment of large language models (LLMs). We present the first systematic study linking hallucination incidence to internal-state drift induced by incremental context injection. Using TruthfulQA, we construct two 16-round "titration" tracks per question: one appends relevant but partially flawed snippets, the other injects deliberately misleading content. Across six open-source LLMs, we track overt hallucination rates with a tri-perspective detector and covert dynamics via cosine, entropy, JS and Spearman drifts of hidden states and attention maps. Results reveal (1) monotonic growth of hallucination frequency and representation drift that plateaus after 5--7 rounds; (2) relevant context drives deeper semantic assimilation, producing high-confidence "self-consistent" hallucinations, whereas irrelevant context induces topic-drift errors anchored by attention re-routing; and (3) convergence of JS-Drift ($\sim0.69$) and Spearman-Drift ($\sim0$) marks an "attention-locking" threshold beyond which hallucinations solidify and become resistant to correction. Correlation analyses expose a seesaw between assimilation capacity and attention diffusion, clarifying size-dependent error modes. These findings supply empirical foundations for intrinsic hallucination prediction and context-aware mitigation mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16888v1">CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04380v2">Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 33 pages, 20 figures, 21 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this work, we investigate the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-design for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16831v1">Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 44 pages
    </div>
    <details class="paper-abstract">
      Unlearning in large language models (LLMs) is intended to remove the influence of specific data, yet current evaluations rely heavily on token-level metrics such as accuracy and perplexity. We show that these metrics can be misleading: models often appear to forget, but their original behavior can be rapidly restored with minimal fine-tuning, revealing that unlearning may obscure information rather than erase it. To diagnose this phenomenon, we introduce a representation-level evaluation framework using PCA-based similarity and shift, centered kernel alignment, and Fisher information. Applying this toolkit across six unlearning methods, three domains (text, code, math), and two open-source LLMs, we uncover a critical distinction between reversible and irreversible forgetting. In reversible cases, models suffer token-level collapse yet retain latent features; in irreversible cases, deeper representational damage occurs. We further provide a theoretical account linking shallow weight perturbations near output layers to misleading unlearning signals, and show that reversibility is modulated by task type and hyperparameters. Our findings reveal a fundamental gap in current evaluation practices and establish a new diagnostic foundation for trustworthy unlearning in LLMs. We provide a unified toolkit for analyzing LLM representation changes under unlearning and relearning: https://github.com/XiaoyuXU1/Representational_Analysis_Tools.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16765v1">When Safety Detectors Aren't Enough: A Stealthy and Effective Jailbreak Attack on LLMs via Steganographic Techniques</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Jailbreak attacks pose a serious threat to large language models (LLMs) by bypassing built-in safety mechanisms and leading to harmful outputs. Studying these attacks is crucial for identifying vulnerabilities and improving model security. This paper presents a systematic survey of jailbreak methods from the novel perspective of stealth. We find that existing attacks struggle to simultaneously achieve toxic stealth (concealing toxic content) and linguistic stealth (maintaining linguistic naturalness). Motivated by this, we propose StegoAttack, a fully stealthy jailbreak attack that uses steganography to hide the harmful query within benign, semantically coherent text. The attack then prompts the LLM to extract the hidden query and respond in an encrypted manner. This approach effectively hides malicious intent while preserving naturalness, allowing it to evade both built-in and external safety mechanisms. We evaluate StegoAttack on four safety-aligned LLMs from major providers, benchmarking against eight state-of-the-art methods. StegoAttack achieves an average attack success rate (ASR) of 92.00%, outperforming the strongest baseline by 11.0%. Its ASR drops by less than 1% even under external detection (e.g., Llama Guard). Moreover, it attains the optimal comprehensive scores on stealth detection metrics, demonstrating both high efficacy and exceptional stealth capabilities. The code is available at https://anonymous.4open.science/r/StegoAttack-Jail66
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16674v2">Through the LLM Looking Glass: A Socratic Probing of Donkeys, Elephants, and Markets</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      While detecting and avoiding bias in LLM-generated text is becoming increasingly important, media bias often remains subtle and subjective, making it particularly difficult to identify and mitigate. In this study, we assess media bias in LLM-generated content and LLMs' ability to detect subtle ideological bias. We conduct this evaluation using two datasets, PoliGen and EconoLex, covering political and economic discourse, respectively. We evaluate seven widely used LLMs by prompting them to generate articles and analyze their ideological preferences via Socratic probing. By using our self-contained Socratic approach, the study aims to directly measure the models' biases rather than relying on external interpretations, thereby minimizing subjective judgments about media bias. Our results reveal a consistent preference of Democratic over Republican positions across all models. Conversely, in economic topics, biases vary among Western LLMs, while those developed in China lean more strongly toward socialism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15091v2">ThinkRec: Thinking-based recommendation via LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled more semantic-aware recommendations through natural language generation. Existing LLM for recommendation (LLM4Rec) methods mostly operate in a System 1-like manner, relying on superficial features to match similar items based on click history, rather than reasoning through deeper behavioral logic. This often leads to superficial and erroneous recommendations. Motivated by this, we propose ThinkRec, a thinking-based framework that shifts LLM4Rec from System 1 to System 2 (rational system). Technically, ThinkRec introduces a thinking activation mechanism that augments item metadata with keyword summarization and injects synthetic reasoning traces, guiding the model to form interpretable reasoning chains that consist of analyzing interaction histories, identifying user preferences, and making decisions based on target items. On top of this, we propose an instance-wise expert fusion mechanism to reduce the reasoning difficulty. By dynamically assigning weights to expert models based on users' latent features, ThinkRec adapts its reasoning path to individual users, thereby enhancing precision and personalization. Extensive experiments on real-world datasets demonstrate that ThinkRec significantly improves the accuracy and interpretability of recommendations. Our implementations are available in anonymous Github: https://github.com/Yu-Qi-hang/ThinkRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16737v1">Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      The significant progress of large language models (LLMs) has led to remarkable achievements across numerous applications. However, their ability to generate harmful content has sparked substantial safety concerns. Despite the implementation of safety alignment techniques during the pre-training phase, recent research indicates that fine-tuning LLMs on adversarial or even benign data can inadvertently compromise their safety. In this paper, we re-examine the fundamental issue of why fine-tuning on non-harmful data still results in safety degradation. We introduce a safety-aware probing (SAP) optimization framework designed to mitigate the safety risks of fine-tuning LLMs. Specifically, SAP incorporates a safety-aware probe into the gradient propagation process, mitigating the model's risk of safety degradation by identifying potential pitfalls in gradient directions, thereby enhancing task-specific performance while successfully preserving model safety. Our extensive experimental results demonstrate that SAP effectively reduces harmfulness below the original fine-tuned model and achieves comparable test loss to standard fine-tuning methods. Our code is available at https://github.com/ChengcanWu/SAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01822v4">Firewalls to Secure Dynamic LLM Agentic Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      LLM agents will likely communicate on behalf of users with other entity-representing agents on tasks involving long-horizon plans with interdependent goals. Current work neglects these agentic networks and their challenges. We identify required properties for agent communication: proactivity, adaptability, privacy (sharing only task-necessary information), and security (preserving integrity and utility against selfish entities). After demonstrating communication vulnerabilities, we propose a practical design and protocol inspired by network security principles. Our framework automatically derives task-specific rules from prior conversations to build firewalls. These firewalls construct a closed language that is completely controlled by the developer. They transform any personal data to the allowed degree of permissibility entailed by the task. Both operations are completely quarantined from external attackers, disabling the potential for prompt injections, jailbreaks, or manipulation. By incorporating rules learned from their previous mistakes, agents rewrite their instructions and self-correct during communication. Evaluations on diverse attacks demonstrate our framework significantly reduces privacy and security vulnerabilities while allowing adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16710v1">Training Long-Context LLMs Efficiently via Chunk-wise Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      While long-context large language models (LLMs) exhibit remarkable document processing capabilities, their prohibitively high training costs often hinder customized applications. To mitigate this issue, we propose \textit{Sequential Chunk-wise Optimization} (SeCO), a memory-efficient training paradigm that partitions lengthy inputs into manageable chunks. Each chunk independently constructs its computational graph and performs localized backpropagation, ensuring that only one chunk's forward activations are stored in memory. Building on SeCO, we further introduce \textit{Sparse Chunk-wise Optimization} (SpaCO), which reduces computational overhead by selectively propagating gradients to specific chunks and incorporates a carefully designed compensation factor to ensure unbiased gradient estimation. SpaCO decouples the computational cost of backpropagation from the context length, enabling training time to gradually converge to inference time as sequences become longer. Implemented as lightweight training wrappers, both SeCO and SpaCO offer substantial practical benefits. For example, when fine-tuning an 8B model with LoRA on a single RTX 3090 GPU, SeCO expands maximum sequence length from 1K to 16K tokens, while SpaCO demonstrates accelerated training speed -- achieving up to 3x faster than SeCO under the same experimental setup. These innovations provide new insights into optimizing long-context models, making them more accessible for practical applications. We have open-sourced the code at \href{https://github.com/wenhaoli-xmu/seco}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16703v1">Locate-then-Merge: Neuron-Level Parameter Fusion for Mitigating Catastrophic Forgetting in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Although multimodal large language models (MLLMs) have achieved impressive performance, the multimodal instruction tuning stage often causes catastrophic forgetting of the base LLM's language ability, even in strong models like Llama3. To address this, we propose Locate-then-Merge, a training-free parameter fusion framework that first locates important parameters and then selectively merges them. We further introduce Neuron-Fusion, a neuron-level strategy that preserves the influence of neurons with large parameter shifts--neurons likely responsible for newly acquired visual capabilities--while attenuating the influence of neurons with smaller changes that likely encode general-purpose language skills. This design enables better retention of visual adaptation while mitigating language degradation. Experiments on 13 benchmarks across both language and visual tasks show that Neuron-Fusion consistently outperforms existing model merging methods. Further analysis reveals that our method effectively reduces context hallucination in generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16697v1">Software Architecture Meets LLMs: A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are used for many different software engineering tasks. In software architecture, they have been applied to tasks such as classification of design decisions, detection of design patterns, and generation of software architecture design from requirements. However, there is little overview on how well they work, what challenges exist, and what open problems remain. In this paper, we present a systematic literature review on the use of LLMs in software architecture. We analyze 18 research articles to answer five research questions, such as which software architecture tasks LLMs are used for, how much automation they provide, which models and techniques are used, and how these approaches are evaluated. Our findings show that while LLMs are increasingly applied to a variety of software architecture tasks and often outperform baselines, some areas, such as generating source code from architectural design, cloud-native computing and architecture, and checking conformance remain underexplored. Although current approaches mostly use simple prompting techniques, we identify a growing research interest in refining LLM-based approaches by integrating advanced techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16690v1">Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $\tau$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $\tau$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16667v1">ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 ACL 2025 Main. Our code and dataset are available at https://github.com/SCUNLP/ELABORATION
    </div>
    <details class="paper-abstract">
      While recent research increasingly emphasizes the value of human-LLM collaboration in competitive programming and proposes numerous empirical methods, a comprehensive understanding remains elusive due to the fragmented nature of existing studies and their use of diverse, application-specific human feedback. Thus, our work serves a three-fold purpose: First, we present the first taxonomy of human feedback consolidating the entire programming process, which promotes fine-grained evaluation. Second, we introduce ELABORATIONSET, a novel programming dataset specifically designed for human-LLM collaboration, meticulously annotated to enable large-scale simulated human feedback and facilitate costeffective real human interaction studies. Third, we introduce ELABORATION, a novel benchmark to facilitate a thorough assessment of human-LLM competitive programming. With ELABORATION, we pinpoint strengthes and weaknesses of existing methods, thereby setting the foundation for future improvement. Our code and dataset are available at https://github.com/SCUNLP/ELABORATION
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16646v1">SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large Language Models have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine mathematical reasoning or superficial pattern recognition. Common evaluation metrics, such as final answer accuracy, fail to disentangle the underlying competencies involved, offering limited diagnostic value. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework. SMART decomposes mathematical problem solving into four distinct dimensions: understanding, reasoning, arithmetic, and reflection \& refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. Crucially, SMART integrates an automated self-generating and self-validating mechanism to produce and verify benchmark data, ensuring both scalability and reliability. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings demonstrate the inadequacy of final answer accuracy as a sole metric and motivate a new holistic metric to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13865v3">Breaking Information Cocoons: A Hyperbolic Graph-LLM Framework for Exploration and Exploitation in Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Modern recommender systems often create information cocoons, restricting users' exposure to diverse content. A key challenge lies in balancing content exploration and exploitation while allowing users to adjust their recommendation preferences. Intuitively, this balance can be modeled as a tree-structured representation, where depth search facilitates exploitation and breadth search enables exploration. However, existing approaches face two fundamental limitations: Euclidean methods struggle to capture hierarchical structures, while hyperbolic methods, despite their superior hierarchical modeling, lack semantic understanding of user and item profiles and fail to provide a principled mechanism for balancing exploration and exploitation. To address these challenges, we propose HERec, a hyperbolic graph-LLM framework that effectively balances exploration and exploitation in recommender systems. Our framework introduces two key innovations: (1) a semantic-enhanced hierarchical mechanism that aligns rich textual descriptions processed by large language models (LLMs) with collaborative information directly in hyperbolic space, allowing for more nuanced updates that respect the underlying hierarchical structure in user-item profiles; (2) an automatic hierarchical representation by optimizing Dasgupta's cost, which discovers hierarchical structures without requiring predefined hyperparameters, enabling user-adjustable exploration-exploitation trade-offs. Extensive experiments demonstrate that HERec consistently outperforms both Euclidean and hyperbolic baselines, achieving up to 5.49% improvement in utility metrics and 11.39% increase in diversity metrics, effectively mitigating information cocoons. We open-source our model implementation at https://github.com/Martin-qyma/HERec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02172v2">Identifying Legal Holdings with LLMs: A Systematic Study of Performance, Scale, and Memorization</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Presented as a short paper at International Conference on Artificial Intelligence and Law 2025 (Chicago, IL)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance in capabilities, it is essential to assess how they perform on established benchmarks. In this study, we present a suite of experiments to assess the performance of modern LLMs (ranging from 3B to 90B+ parameters) on CaseHOLD, a legal benchmark dataset for identifying case holdings. Our experiments demonstrate ``scaling effects'' - performance on this task improves with model size, with more capable models like GPT4o and AmazonNovaPro achieving macro F1 scores of 0.744 and 0.720 respectively. These scores are competitive with the best published results on this dataset, and do not require any technically sophisticated model training, fine-tuning or few-shot prompting. To ensure that these strong results are not due to memorization of judicial opinions contained in the training data, we develop and utilize a novel citation anonymization test that preserves semantic meaning while ensuring case names and citations are fictitious. Models maintain strong performance under these conditions (macro F1 of 0.728), suggesting the performance is not due to rote memorization. These findings demonstrate both the promise and current limitations of LLMs for legal tasks with important implications for the development and measurement of automated legal analytics and legal benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10063v2">Hallucination Detection in LLMs with Topological Divergence on Attention Graphs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments - including evaluation on question answering and summarization tasks - show that our approach achieves state-of-the-art or competitive results on several benchmarks while requiring minimal annotated data and computational resources. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10347v3">A Unified Approach to Routing and Cascading for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      The availability of a wide range of large language models (LLMs) embedded in various agentic systems has significantly increased the potential of model selection strategies to improve the cost-performance tradeoff. Existing strategies involve either routing, where a single model is chosen per query, or cascading, which sequentially runs increasingly larger models until a satisfactory answer is found. However, current approaches face three key limitations: they (1) lack formal proofs of optimality, (2) fail to identify the conditions under which these strategies are most effective to improve the cost-performance tradeoff, and (3) are unable to combine both paradigms for further improvements. To address these issues, we first derive a novel optimal strategy for cascading and prove the optimality of an existing routing strategy. Further, we propose cascade routing, a unified framework that integrates routing and cascading into a theoretically optimal strategy. Through our analysis, we identify good quality estimators as the critical factor for the success of model selection paradigms. Finally, in our experiments, we show that cascade routing consistently outperforms the individual approaches by a large margin and we analyze quality estimators to determine when routing and/or cascading are useful paradigms for model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16590v1">Beyond LLMs: An Exploration of Small Open-source Language Models in Logging Statement Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Effective software maintenance heavily relies on high-quality logging statements, but manual logging is challenging, error-prone, and insufficiently standardized, often leading to inconsistent log quality. While large language models have shown promise in automatic logging, they introduce concerns regarding privacy, resource intensity, and adaptability to specific enterprise needs. To tackle these limitations, this paper empirically investigates whether Small Open-source Language Models (SOLMs) could become a viable alternative via proper exploitation. Specifically, we conduct a large-scale empirical study on four prominent SOLMs, systematically evaluating the impacts of various interaction strategies, parameter-efficient fine-tuning techniques, model sizes, and model types in automatic logging. Our key findings reveal that Retrieval-Augmented Generation significantly enhances performance, and LoRA is a highly effective PEFT technique. While larger SOLMs tend to perform better, this involves a trade-off with computational resources, and instruct-tuned SOLMs generally surpass their base counterparts. Notably, fine-tuned SOLMs, particularly Qwen2.5-coder-14B, outperformed existing specialized tools and LLM baselines in accurately predicting logging locations and generating high-quality statements, a conclusion supported by traditional evaluation metrics and LLM-as-a-judge evaluations. Furthermore, SOLMs also demonstrated robust generalization across diverse, unseen code repositories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16570v1">URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are commonly pretrained on vast corpora of text without utilizing contextual metadata such as source, quality, or topic, leading to a context-free learning paradigm. While recent studies suggest that adding metadata like URL information as context (i.e., auxiliary inputs not used in the loss calculation) can improve training efficiency and downstream performance, they offer limited understanding of which types of metadata are truly effective and under what conditions. In this work, we conduct a systematic evaluation and find that not all metadata types contribute equally. Only URL context speeds up training, whereas quality scores and topic/format domain information offer no clear benefit. Furthermore, the improved downstream performances of URL conditioning emerge only when longer prompts are used at inference time. In addition, we demonstrate that context-aware pretraining enables more controllable generation than context-free pretraining, in a classifier-free guidance fashion. Although topic and format metadata do not accelerate training, they are effective for steering outputs, offering human-interpretable control over generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16567v1">Finetuning-Activated Backdoors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16552v1">Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are computationally expensive and inefficient. In this paper, we introduce Compressed Latent Reasoning (CoLaR), a novel framework that dynamically compresses reasoning processes in latent space through a two-stage training approach. First, during supervised fine-tuning, CoLaR extends beyond next-token prediction by incorporating an auxiliary next compressed embedding prediction objective. This process merges embeddings of consecutive tokens using a compression factor randomly sampled from a predefined range, and trains a specialized latent head to predict distributions of subsequent compressed embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that leverages the latent head's non-deterministic nature to explore diverse reasoning paths and exploit more compact ones. This approach enables CoLaR to: i) perform reasoning at a dense latent level (i.e., silently), substantially reducing reasoning chain length, and ii) dynamically adjust reasoning speed at inference time by simply prompting the desired compression factor. Extensive experiments across four mathematical reasoning datasets demonstrate that CoLaR achieves 14.1% higher accuracy than latent-based baseline methods at comparable compression ratios, and reduces reasoning chain length by 53.3% with only 4.8% performance degradation compared to explicit CoT method. Moreover, when applied to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR demonstrates performance gains of up to 5.4% while dramatically reducing latent reasoning chain length by 82.8%. The code and models will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16530v1">DuFFin: A Dual-Level Fingerprinting Framework for LLMs IP Protection</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are considered valuable Intellectual Properties (IP) for legitimate owners due to the enormous computational cost of training. It is crucial to protect the IP of LLMs from malicious stealing or unauthorized deployment. Despite existing efforts in watermarking and fingerprinting LLMs, these methods either impact the text generation process or are limited in white-box access to the suspect model, making them impractical. Hence, we propose DuFFin, a novel $\textbf{Du}$al-Level $\textbf{Fin}$gerprinting $\textbf{F}$ramework for black-box setting ownership verification. DuFFin extracts the trigger pattern and the knowledge-level fingerprints to identify the source of a suspect model. We conduct experiments on a variety of models collected from the open-source website, including four popular base models as protected LLMs and their fine-tuning, quantization, and safety alignment versions, which are released by large companies, start-ups, and individual users. Results show that our method can accurately verify the copyright of the base protected LLM on their model variants, achieving the IP-ROC metric greater than 0.95. Our code is available at https://github.com/yuliangyan0807/llm-fingerprint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16522v1">Benchmarking and Pushing the Multi-Bias Elimination Boundary of LLMs via Causal Effect Estimation-guided Debiasing</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Despite significant progress, recent studies have indicated that current large language models (LLMs) may still utilize bias during inference, leading to the poor generalizability of LLMs. Some benchmarks are proposed to investigate the generalizability of LLMs, with each piece of data typically containing one type of controlled bias. However, a single piece of data may contain multiple types of biases in practical applications. To bridge this gap, we propose a multi-bias benchmark where each piece of data contains five types of biases. The evaluations conducted on this benchmark reveal that the performance of existing LLMs and debiasing methods is unsatisfying, highlighting the challenge of eliminating multiple types of biases simultaneously. To overcome this challenge, we propose a causal effect estimation-guided multi-bias elimination method (CMBE). This method first estimates the causal effect of multiple types of biases simultaneously. Subsequently, we eliminate the causal effect of biases from the total causal effect exerted by both the semantic information and biases during inference. Experimental results show that CMBE can effectively eliminate multiple types of bias simultaneously to enhance the generalizability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16520v1">Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Factual hallucinations are a major challenge for Large Language Models (LLMs). They undermine reliability and user trust by generating inaccurate or fabricated content. Recent studies suggest that when generating false statements, the internal states of LLMs encode information about truthfulness. However, these studies often rely on synthetic datasets that lack realism, which limits generalization when evaluating the factual accuracy of text generated by the model itself. In this paper, we challenge the findings of previous work by investigating truthfulness encoding capabilities, leading to the generation of a more realistic and challenging dataset. Specifically, we extend previous work by introducing: (1) a strategy for sampling plausible true-false factoid sentences from tabular data and (2) a procedure for generating realistic, LLM-dependent true-false datasets from Question Answering collections. Our analysis of two open-source LLMs reveals that while the findings from previous studies are partially validated, generalization to LLM-generated datasets remains challenging. This study lays the groundwork for future research on factuality in LLMs and offers practical guidelines for more effective evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16502v1">Recursive Offloading for LLM Serving in Multi-tier Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Heterogeneous device-edge-cloud computing infrastructures have become widely adopted in telecommunication operators and Wide Area Networks (WANs), offering multi-tier computational support for emerging intelligent services. With the rapid proliferation of Large Language Model (LLM) services, efficiently coordinating inference tasks and reducing communication overhead within these multi-tier network architectures becomes a critical deployment challenge. Existing LLM serving paradigms exhibit significant limitations: on-device deployment supports only lightweight LLMs due to hardware constraints, while cloud-centric deployment suffers from resource congestion and considerable prompt communication overhead caused by frequent service requests during peak periods. Although the model-cascading-based inference strategy adapts better to multi-tier networks, its reliance on fine-grained, manually adjusted thresholds makes it less responsive to dynamic network conditions and varying task complexities. To address these challenges, we propose RecServe, a recursive offloading framework tailored for LLM serving in multi-tier networks. RecServe integrates a task-specific hierarchical confidence evaluation mechanism that guides offloading decisions based on inferred task complexity in progressively scaled LLMs across device, edge, and cloud tiers. To further enable intelligent task routing across tiers, RecServe employs a sliding-window-based dynamic offloading strategy with quantile interpolation, enabling real-time tracking of historical confidence distributions and adaptive offloading threshold adjustments. Experiments on eight datasets demonstrate that RecServe outperforms CasServe in both service quality and communication efficiency, and reduces the communication burden by over 50\% compared to centralized cloud-based serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15701v2">When LLMs Learn to be Students: The SOEI Framework for Modeling and Evaluating Virtual Student Agents in Educational Interaction</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled intelligent tutoring systems, yet the development of LLM-based Virtual Student Agents (LVSAs) remains underexplored. Such agents are essential for teacher-facing applications, where simulating diverse learner traits can support adaptive instruction and pedagogical skill development. However, current methods lack principled personality modeling, scalable evaluation of behavioral consistency, and empirical validation in interactive teaching settings. We propose the SOEI framework, a structured pipeline comprising Scene, Object, Evaluation, and Interaction, for constructing and evaluating personality-aligned LVSAs in classroom scenarios. Leveraging Chinese language instruction as a cognitively and emotionally rich testbed, we generate five LVSAs based on Big Five traits through LoRA fine-tuning and expert-informed prompt design. Their behavioral realism and personality coherence are assessed using a hybrid human & GPT-4 evaluation and a multi-dimensional annotation protocol. Through controlled experiments with real pre-service teachers, we demonstrate that LVSAs can elicit adaptive teaching strategies and maintain trait-consistent behavior across multi-turn dialogues. Our results provide: (1) an educationally and psychologically grounded generation pipeline for LLM-based student agents; (2) a hybrid, scalable evaluation framework for behavioral realism; and (3) empirical insights into the pedagogical utility of LVSAs in shaping instructional adaptation. By embedding LVSAs into both generative modeling and human-in-the-loop teaching, SOEI bridges AI for Education (AI4Edu) and Education for AI (Edu4AI), positioning classroom interaction as a rigorous testbed for controllability, personality alignment, and human-likeness in large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05423v3">LiTransProQA: an LLM-based Literary Translation evaluation metric with Professional Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Updated version, with examples in the appendix
    </div>
    <details class="paper-abstract">
      The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation as being superior to human translation from experienced professionals. In the long run, this bias could result in an irreversible decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce LiTransProQA, a novel, reference-free, LLM-based question-answering framework designed for literary translation evaluation. LiTransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, LiTransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation and surpassing the best state-of-the-art metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, LiTransProQA reaches human-level evaluation performance comparable to trained student evaluators. It shows broad applicability to open-source models like LLaMa3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free tool for evaluating literary translations that require local processing due to copyright or ethical considerations. The code and datasets are available under: https://github.com/zhangr2021/TransProQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16467v1">Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs) infer user's demographic information from subtle cues in the conversation -- a phenomenon called implicit personalization. Prior work has shown that such inferences can lead to lower quality responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model's internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06846v2">Prot2Chat: Protein LLM with Early-Fusion of Text, Sequence and Structure</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Motivation: Proteins are of great significance in living organisms. However, understanding their functions encounters numerous challenges, such as insufficient integration of multimodal information, a large number of training parameters, limited flexibility of classification-based methods, and the lack of systematic evaluation metrics for protein Q&A systems. To tackle these issues, we propose the Prot2Chat framework. Results: We modified ProteinMPNN to encode protein sequence and structural information in a unified way. We used a large language model (LLM) to encode questions into vectors and developed a protein-text adapter to compress protein information into virtual tokens based on these vectors, achieving the early fusion of text and protein information. Finally, the same LLM reads the virtual tokens and the questions to generate answers. To optimize training efficiency, we froze the encoder and employed Low-Rank Adaptation (LoRA) techniques for the LLM. Experiments on two datasets show that both automated metrics and expert evaluations demonstrate the superior performance of our model, and zero-shot prediction results highlight its generalization ability. The models and codes are available at https://github.com/ wangzc1233/Prot2Chat. Contact: zqcao@suda.edu.cn or wangzc025@163.com Key words: Protein Q&A, Early-Fusion, LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11617v2">ASMA-Tune: Unlocking LLMs' Assembly Code Comprehension via Structural-Semantic Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 9 pages, multiple figures
    </div>
    <details class="paper-abstract">
      Assembly code analysis and comprehension play critical roles in applications like reverse engineering, yet they face substantial challenges due to low information density and a lack of explicit syntactic structures. While traditional masked language modeling (MLM) approaches do not explicitly focus on natural language interaction, emerging decoder-focused large language models (LLMs) demonstrate partial success in binary analysis yet remain underexplored for holistic comprehension. We present Assembly Augmented Tuning, an end-to-end structural-semantic instruction tuning framework that synergizes encoder architecture with decoder-based LLMs through a projector module, where the assembly encoder extracts hardware-level structural features, the projector bridges representations with the semantic space, and the instruction-tuned LLM preserves natural language capabilities. Experimental results demonstrate three key advantages: (1) State-of-the-art performance in assembly comprehension with +39.7% Recall@1 and +17.8% MRR improvements over GPT-4-Turbo, (2) Consistent enhancements across base models (24.6-107.4% Recall@1 and 15.2-106.3% MRR on Qwen2.5-Coder, Deepseek-Coder and CodeLlama variants), and (3) Superior instruction-following capabilities (41.5%-118% improvements) with controlled code generation degradation (-8.9% to -35% across architectures).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16456v1">MAGIC: Motion-Aware Generative Inference via Confidence-Guided LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Recent advances in static 3D generation have intensified the demand for physically consistent dynamic 3D content. However, existing video generation models, including diffusion-based methods, often prioritize visual realism while neglecting physical plausibility, resulting in implausible object dynamics. Prior approaches for physics-aware dynamic generation typically rely on large-scale annotated datasets or extensive model fine-tuning, which imposes significant computational and data collection burdens and limits scalability across scenarios. To address these challenges, we present MAGIC, a training-free framework for single-image physical property inference and dynamic generation, integrating pretrained image-to-video diffusion models with iterative LLM-based reasoning. Our framework generates motion-rich videos from a static image and closes the visual-to-physical gap through a confidence-driven LLM feedback loop that adaptively steers the diffusion model toward physics-relevant motion. To translate visual dynamics into controllable physical behavior, we further introduce a differentiable MPM simulator operating directly on 3D Gaussians reconstructed from the single image, enabling physically grounded, simulation-ready outputs without any supervision or model tuning. Experiments show that MAGIC outperforms existing physics-aware generative methods in inference accuracy and achieves greater temporal coherence than state-of-the-art video diffusion models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16455v1">Psychology-driven LLM Agents for Explainable Panic Prediction on Social Media during Sudden Disaster Events</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      During sudden disaster events, accurately predicting public panic sentiment on social media is crucial for proactive governance and crisis management. Current efforts on this problem face three main challenges: lack of finely annotated data hinders emotion prediction studies, unmodeled risk perception causes prediction inaccuracies, and insufficient interpretability of panic formation mechanisms. We address these issues by proposing a Psychology-driven generative Agent framework (PsychoAgent) for explainable panic prediction based on emotion arousal theory. Specifically, we first construct a fine-grained open panic emotion dataset (namely COPE) via human-large language models (LLMs) collaboration to mitigate semantic bias. Then, we develop a framework integrating cross-domain heterogeneous data grounded in psychological mechanisms to model risk perception and cognitive differences in emotion generation. To enhance interpretability, we design an LLM-based role-playing agent that simulates individual psychological chains through dedicatedly designed prompts. Experimental results on our annotated dataset show that PsychoAgent improves panic emotion prediction performance by 12.6% to 21.7% compared to baseline models. Furthermore, the explainability and generalization of our approach is validated. Crucially, this represents a paradigm shift from opaque "data-driven fitting" to transparent "role-based simulation with mechanistic interpretation" for panic emotion prediction during emergencies. Our implementation is publicly available at: https://anonymous.4open.science/r/PsychoAgent-19DD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16410v1">Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have shown remarkable reasoning capabilities via large-scale reinforcement learning (RL). However, leveraging the RL algorithm to empower effective multi-tool collaborative reasoning in LLMs remains an open challenge. In this paper, we introduce Tool-Star, an RL-based framework designed to empower LLMs to autonomously invoke multiple external tools during stepwise reasoning. Tool-Star integrates six types of tools and incorporates systematic designs in both data synthesis and training. To address the scarcity of tool-use data, we propose a general tool-integrated reasoning data synthesis pipeline, which combines tool-integrated prompting with hint-based sampling to automatically and scalably generate tool-use trajectories. A subsequent quality normalization and difficulty-aware classification process filters out low-quality samples and organizes the dataset from easy to hard. Furthermore, we propose a two-stage training framework to enhance multi-tool collaborative reasoning by: (1) cold-start fine-tuning, which guides LLMs to explore reasoning patterns via tool-invocation feedback; and (2) a multi-tool self-critic RL algorithm with hierarchical reward design, which reinforces reward understanding and promotes effective tool collaboration. Experimental analyses on over 10 challenging reasoning benchmarks highlight the effectiveness and efficiency of Tool-Star. The code is available at https://github.com/dongguanting/Tool-Star.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16408v1">From Surveys to Narratives: Rethinking Cultural Value Adaptation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Adapting cultural values in Large Language Models (LLMs) presents significant challenges, particularly due to biases and limited training data. Prior work primarily aligns LLMs with different cultural values using World Values Survey (WVS) data. However, it remains unclear whether this approach effectively captures cultural nuances or produces distinct cultural representations for various downstream tasks. In this paper, we systematically investigate WVS-based training for cultural value adaptation and find that relying solely on survey data can homogenize cultural norms and interfere with factual knowledge. To investigate these issues, we augment WVS with encyclopedic and scenario-based cultural narratives from Wikipedia and NormAd. While these narratives may have variable effects on downstream tasks, they consistently improve cultural distinctiveness than survey data alone. Our work highlights the inherent complexity of aligning cultural values with the goal of guiding task-specific behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06086v2">Is a Peeled Apple Still Red? Evaluating LLMs' Ability for Conceptual Combination with Property Type</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 NAACL 2025 Oral
    </div>
    <details class="paper-abstract">
      Conceptual combination is a cognitive process that merges basic concepts, enabling the creation of complex expressions. During this process, the properties of combination (e.g., the whiteness of a peeled apple) can be inherited from basic concepts, newly emerge, or be canceled. However, previous studies have evaluated a limited set of properties and have not examined the generative process. To address this gap, we introduce the Conceptual Combination with Property Type dataset (CCPT), which consists of 12.3K annotated triplets of noun phrases, properties, and property types. Using CCPT, we establish three types of tasks to evaluate LLMs for conceptual combination thoroughly. Our key findings are threefold: (1) Our automatic metric grading property emergence and cancellation closely corresponds with human judgments. (2) LLMs, including OpenAI's o1, struggle to generate noun phrases which possess given emergent properties. (3) Our proposed method, inspired by cognitive psychology model that explains how relationships between concepts are formed, improves performances in all generative tasks. The dataset and experimental code are available at https://github.com/seokwon99/CCPT.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16363v1">AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      We introduce AdamS, a simple yet effective alternative to Adam for large language model (LLM) pretraining and post-training. By leveraging a novel denominator, i.e., the root of weighted sum of squares of the momentum and the current gradient, AdamS eliminates the need for second-moment estimates. Hence, AdamS is efficient, matching the memory and compute footprint of SGD with momentum while delivering superior optimization performance. Moreover, AdamS is easy to adopt: it can directly inherit hyperparameters of AdamW, and is entirely model-agnostic, integrating seamlessly into existing pipelines without modifications to optimizer APIs or architectures. The motivation behind AdamS stems from the observed $(L_0, L_1)$ smoothness properties in transformer objectives, where local smoothness is governed by gradient magnitudes that can be further approximated by momentum magnitudes. We establish rigorous theoretical convergence guarantees and provide practical guidelines for hyperparameter selection. Empirically, AdamS demonstrates strong performance in various tasks, including pre-training runs on GPT-2 and Llama2 (up to 13B parameters) and reinforcement learning in post-training regimes. With its efficiency, simplicity, and theoretical grounding, AdamS stands as a compelling alternative to existing optimizers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16340v1">Improving Chemical Understanding of LLMs via SMILES Parsing</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly recognized as powerful tools for scientific discovery, particularly in molecular science. A fundamental requirement for these models is the ability to accurately understand molecular structures, commonly encoded in the SMILES representation. However, current LLMs struggle to interpret SMILES, even failing to carry out basic tasks such as counting molecular rings. To address this limitation, we introduce CLEANMOL, a novel framework that formulates SMILES parsing into a suite of clean and deterministic tasks explicitly designed to promote graph-level molecular comprehension. These tasks span from subgraph matching to global graph matching, providing structured supervision aligned with molecular structural properties. We construct a molecular pretraining dataset with adaptive difficulty scoring and pre-train open-source LLMs on these tasks. Our results show that CLEANMOL not only enhances structural comprehension but also achieves the best or competes with the baseline on the Mol-Instructions benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16339v1">Rethinking Code Review Workflows with LLM Assistance: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Code reviews are a critical yet time-consuming aspect of modern software development, increasingly challenged by growing system complexity and the demand for faster delivery. This paper presents a study conducted at WirelessCar Sweden AB, combining an exploratory field study of current code review practices with a field experiment involving two variations of an LLM-assisted code review tool. The field study identifies key challenges in traditional code reviews, including frequent context switching, insufficient contextual information, and highlights both opportunities (e.g., automatic summarization of complex pull requests) and concerns (e.g., false positives and trust issues) in using LLMs. In the field experiment, we developed two prototype variations: one offering LLM-generated reviews upfront and the other enabling on-demand interaction. Both utilize a semantic search pipeline based on retrieval-augmented generation to assemble relevant contextual information for the review, thereby tackling the uncovered challenges. Developers evaluated both variations in real-world settings: AI-led reviews are overall more preferred, while still being conditional on the reviewers' familiarity with the code base, as well as on the severity of the pull request.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16312v1">EquivPruner: Boosting Efficiency and Quality in LLM-Based Search via Action Pruning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at complex reasoning through search algorithms, yet current strategies often suffer from massive token consumption due to redundant exploration of semantically equivalent steps. Existing semantic similarity methods struggle to accurately identify such equivalence in domain-specific contexts like mathematical reasoning. To address this, we propose EquivPruner, a simple yet effective approach that identifies and prunes semantically equivalent actions during LLM reasoning search. We also introduce MathEquiv, the first dataset we created for mathematical statement equivalence, which enables the training of a lightweight equivalence detector. Extensive experiments across various models and tasks demonstrate that EquivPruner significantly reduces token consumption, improving searching efficiency and often bolstering reasoning accuracy. For instance, when applied to Qwen2.5-Math-7B-Instruct on GSM8K, EquivPruner reduced token consumption by 48.1\% while also improving accuracy. Our code is available at https://github.com/Lolo1222/EquivPruner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11471v3">GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted by ACL2025(Findings)
    </div>
    <details class="paper-abstract">
      Knowledge Graph Completion (KGC), which aims to infer missing or incomplete facts, is a crucial task for KGs. However, integrating the vital structural information of KGs into Large Language Models (LLMs) and outputting predictions deterministically remains challenging. To address this, we propose a new method called GLTW, which encodes the structural information of KGs and merges it with LLMs to enhance KGC performance. Specifically, we introduce an improved Graph Transformer (iGT) that effectively encodes subgraphs with both local and global structural information and inherits the characteristics of language model, bypassing training from scratch. Also, we develop a subgraph-based multi-classification training objective, using all entities within KG as classification objects, to boost learning efficiency.Importantly, we combine iGT with an LLM that takes KG language prompts as input.Our extensive experiments on various KG datasets show that GLTW achieves significant performance gains compared to SOTA baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16303v1">INFERENCEDYNAMICS: Efficient Routing Across LLMs through Structured Capability and Knowledge Profiling</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) routing is a pivotal technique for navigating a diverse landscape of LLMs, aiming to select the best-performing LLMs tailored to the domains of user queries, while managing computational resources. However, current routing approaches often face limitations in scalability when dealing with a large pool of specialized LLMs, or in their adaptability to extending model scope and evolving capability domains. To overcome those challenges, we propose InferenceDynamics, a flexible and scalable multi-dimensional routing framework by modeling the capability and knowledge of models. We operate it on our comprehensive dataset RouteMix, and demonstrate its effectiveness and generalizability in group-level routing using modern benchmarks including MMLU-Pro, GPQA, BigGenBench, and LiveBench, showcasing its ability to identify and leverage top-performing models for given tasks, leading to superior outcomes with efficient resource utilization. The broader adoption of Inference Dynamics can empower users to harness the full specialized potential of the LLM ecosystem, and our code will be made publicly available to encourage further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16293v1">Augmenting LLM Reasoning with Dynamic Notes Writing for Complex QA</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Iterative RAG for multi-hop question answering faces challenges with lengthy contexts and the buildup of irrelevant information. This hinders a model's capacity to process and reason over retrieved content and limits performance. While recent methods focus on compressing retrieved information, they are either restricted to single-round RAG, require finetuning or lack scalability in iterative RAG. To address these challenges, we propose Notes Writing, a method that generates concise and relevant notes from retrieved documents at each step, thereby reducing noise and retaining only essential information. This indirectly increases the effective context length of Large Language Models (LLMs), enabling them to reason and plan more effectively while processing larger volumes of input text. Notes Writing is framework agnostic and can be integrated with different iterative RAG methods. We demonstrate its effectiveness with three iterative RAG methods, across two models and four evaluation datasets. Notes writing yields an average improvement of 15.6 percentage points overall, with minimal increase in output tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16277v1">Spontaneous Speech Variables for Evaluating LLMs Cognitive Plausibility</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 The 14th Workshop on Cognitive Modeling and Computational Linguistics (CMCL). May 3, 2025. Collocated with NAACL 2025
    </div>
    <details class="paper-abstract">
      The achievements of Large Language Models in Natural Language Processing, especially for high-resource languages, call for a better understanding of their characteristics from a cognitive perspective. Researchers have attempted to evaluate artificial models by testing their ability to predict behavioral (e.g., eye-tracking fixations) and physiological (e.g., brain responses) variables during language processing (e.g., reading/listening). In this paper, we propose using spontaneous speech corpora to derive production variables (speech reductions, prosodic prominences) and applying them in a similar fashion. More precisely, we extract. We then test models trained with a standard procedure on different pretraining datasets (written, spoken, and mixed genres) for their ability to predict these two variables. Our results show that, after some fine-tuning, the models can predict these production variables well above baselines. We also observe that spoken genre training data provides more accurate predictions than written genres. These results contribute to the broader effort of using high-quality speech corpora as benchmarks for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12212v2">Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted by ACL 2025 main, 18 pages, 8 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06139v2">LCIRC: A Recurrent Compression Approach for Efficient Long-form Context and Query Dependent Modeling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted to NAACL 2025. Project Page: https://ssuminan.github.io/LCIRC/
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) excel in generating coherent and contextually rich outputs, their capacity to efficiently handle long-form contexts is limited by fixed-length position embeddings. Additionally, the computational cost of processing long sequences increases quadratically, making it challenging to extend context length. To address these challenges, we propose Long-form Context Injection with Recurrent Compression (LCIRC), a method that enables the efficient processing long-form sequences beyond the model's length limit through recurrent compression without retraining the entire model. We further introduce query dependent context modeling, which selectively compresses query-relevant information, ensuring that the model retains the most pertinent content. Our empirical results demonstrate that Query Dependent LCIRC (QD-LCIRC) significantly improves LLM's ability to manage extended contexts, making it well-suited for tasks that require both comprehensive context understanding and query relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16254v1">Reassessing Collaborative Writing Theories and Frameworks in the Age of LLMs: What Still Applies and What We Must Leave Behind</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      In this paper, we conduct a critical review of existing theories and frameworks on human-human collaborative writing to assess their relevance to the current human-AI paradigm in professional contexts, and draw seven insights along with design implications for human-AI collaborative writing tools. We found that, as LLMs nudge the writing process more towards an empirical "trial and error" process analogous to prototyping, the non-linear cognitive process of writing will stay the same, but more rigor will be required for revision methodologies. This shift would shed further light on the importance of coherence support, but the large language model (LLM)'s unprecedented semantic capabilities can bring novel approaches to this ongoing challenge. We argue that teamwork-related factors such as group awareness, consensus building and authorship - which have been central in human-human collaborative writing studies - should not apply to the human-AI paradigm due to excessive anthropomorphism. With the LLM's text generation capabilities becoming essentially indistinguishable from human-written ones, we are entering an era where, for the first time in the history of computing, we are engaging in collaborative writing with AI at workplaces on a daily basis. We aim to bring theoretical grounding and practical design guidance to the interaction designs of human-AI collaborative writing, with the goal of enhancing future human-AI writing software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16222v1">Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      With the growing use of large language models(LLMs) as evaluators, their application has expanded to code evaluation tasks, where they assess the correctness of generated code without relying on reference implementations. While this offers scalability and flexibility, it also raises a critical, unresolved question: Can LLM judges fairly and robustly evaluate semantically equivalent code with superficial variations? Functionally correct code often exhibits variations-such as differences in variable names, comments, or formatting-that should not influence its correctness. Yet, whether LLM judges can reliably handle these variations remains unclear. We present the first comprehensive study of this issue, defining six types of potential bias in code evaluation and revealing their systematic impact on LLM judges. Across five programming languages and multiple LLMs, we empirically demonstrate that all tested LLM judges are susceptible to both positive and negative biases, resulting in inflated or unfairly low scores. Moreover, we observe that LLM judges remain vulnerable to these biases even when prompted to generate test cases before scoring, highlighting the need for more robust code evaluation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15793v2">HCRMP: A LLM-Hinted Contextual Reinforcement Learning Framework for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) with Reinforcement Learning (RL) can enhance autonomous driving (AD) performance in complex scenarios. However, current LLM-Dominated RL methods over-rely on LLM outputs, which are prone to hallucinations. Evaluations show that state-of-the-art LLM indicates a non-hallucination rate of only approximately 57.95% when assessed on essential driving-related tasks. Thus, in these methods, hallucinations from the LLM can directly jeopardize the performance of driving policies. This paper argues that maintaining relative independence between the LLM and the RL is vital for solving the hallucinations problem. Consequently, this paper is devoted to propose a novel LLM-Hinted RL paradigm. The LLM is used to generate semantic hints for state augmentation and policy optimization to assist RL agent in motion planning, while the RL agent counteracts potential erroneous semantic indications through policy learning to achieve excellent driving performance. Based on this paradigm, we propose the HCRMP (LLM-Hinted Contextual Reinforcement Learning Motion Planner) architecture, which is designed that includes Augmented Semantic Representation Module to extend state space. Contextual Stability Anchor Module enhances the reliability of multi-critic weight hints by utilizing information from the knowledge base. Semantic Cache Module is employed to seamlessly integrate LLM low-frequency guidance with RL high-frequency control. Extensive experiments in CARLA validate HCRMP's strong overall driving performance. HCRMP achieves a task success rate of up to 80.3% under diverse driving conditions with different traffic densities. Under safety-critical driving conditions, HCRMP significantly reduces the collision rate by 11.4%, which effectively improves the driving performance in complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16221v1">LightRouter: Towards Efficient LLM Collaboration with Minimal Overhead</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models has unlocked remarkable capabilities across a diverse array of natural language processing tasks. However, the considerable differences among available LLMs-in terms of cost, performance, and computational demands-pose significant challenges for users aiming to identify the most suitable model for specific tasks. In this work, we present LightRouter, a novel framework designed to systematically select and integrate a small subset of LLMs from a larger pool, with the objective of jointly optimizing both task performance and cost efficiency. LightRouter leverages an adaptive selection mechanism to identify models that require only a minimal number of boot tokens, thereby reducing costs, and further employs an effective integration strategy to combine their outputs. Extensive experiments across multiple benchmarks demonstrate that LightRouter matches or outperforms widely-used ensemble baselines, achieving up to a 25% improvement in accuracy. Compared with leading high-performing models, LightRouter achieves comparable performance while reducing inference costs by up to 27%. Importantly, our framework operates without any prior knowledge of individual models and relies exclusively on inexpensive, lightweight models. This work introduces a practical approach for efficient LLM selection and provides valuable insights into optimal strategies for model combination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13830v2">Improving Noise Robustness of LLM-based Zero-shot TTS via Discrete Acoustic Token Denoising</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted by Interspeech 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based zero-shot text-to-speech (TTS) methods tend to preserve the acoustic environment of the audio prompt, leading to degradation in synthesized speech quality when the audio prompt contains noise. In this paper, we propose a novel neural codec-based speech denoiser and integrate it with the advanced LLM-based TTS model, LauraTTS, to achieve noise-robust zero-shot TTS. The proposed codec denoiser consists of an audio codec, a token denoiser, and an embedding refiner. The token denoiser predicts the first two groups of clean acoustic tokens from the noisy ones, which can serve as the acoustic prompt for LauraTTS to synthesize high-quality personalized speech or be converted to clean speech waveforms through the embedding refiner and codec decoder. Experimental results show that our proposed codec denoiser outperforms state-of-the-art speech enhancement (SE) methods, and the proposed noise-robust LauraTTS surpasses the approach using additional SE models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16216v1">Memorization or Reasoning? Exploring the Idiom Understanding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Idioms have long posed a challenge due to their unique linguistic properties, which set them apart from other common expressions. While recent studies have leveraged large language models (LLMs) to handle idioms across various tasks, e.g., idiom-containing sentence generation and idiomatic machine translation, little is known about the underlying mechanisms of idiom processing in LLMs, particularly in multilingual settings. To this end, we introduce MIDAS, a new large-scale dataset of idioms in six languages, each paired with its corresponding meaning. Leveraging this resource, we conduct a comprehensive evaluation of LLMs' idiom processing ability, identifying key factors that influence their performance. Our findings suggest that LLMs rely not only on memorization, but also adopt a hybrid approach that integrates contextual cues and reasoning, especially when processing compositional idioms. This implies that idiom understanding in LLMs emerges from an interplay between internal knowledge retrieval and reasoning-based inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06272v2">Evaluating LLM-based Approaches to Legal Citation Prediction: Domain-specific Pre-training, Fine-tuning, or RAG? A Benchmark and an Australian Law Case Study</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 For code, data, and models see https://auslawbench.github.io
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong potential across legal tasks, yet the problem of legal citation prediction remains under-explored. At its core, this task demands fine-grained contextual understanding and precise identification of relevant legislation or precedent. We introduce the AusLaw Citation Benchmark, a real-world dataset comprising 55k Australian legal instances and 18,677 unique citations which to the best of our knowledge is the first of its scale and scope. We then conduct a systematic benchmarking across a range of solutions: (i) standard prompting of both general and law-specialised LLMs, (ii) retrieval-only pipelines with both generic and domain-specific embeddings, (iii) supervised fine-tuning, and (iv) several hybrid strategies that combine LLMs with retrieval augmentation through query expansion, voting ensembles, or re-ranking. Results show that neither general nor law-specific LLMs suffice as stand-alone solutions, with performance near zero. Instruction tuning (of even a generic open-source LLM) on task-specific dataset is among the best performing solutions. We highlight that database granularity along with the type of embeddings play a critical role in retrieval-based approaches, with hybrid methods which utilise a trained re-ranker delivering the best results. Despite this, a performance gap of nearly 50% remains, underscoring the value of this challenging benchmark as a rigorous test-bed for future research in legal-domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13862v2">PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09285v2">DynaServe: Unified and Elastic Execution for Dynamic Disaggregated LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      LLM inference must meet strict latency SLOs (e.g., 100 ms P99 time-between-tokens) while maximizing goodput. Yet, real-world variability in prompt and response lengths skews compute-intensive prefill and memory-bound decode phases, making both colocated (even with chunked prefill) and disaggregated deployments unable to simultaneously deliver low tail latency and high throughput. We introduce DynaServe, a high-performance LLM serving system built atop vLLM that unifies and extends both paradigms for maximizing goodput under SLO constraints, when handling unbalanced and dynamic workloads. It relies on a micro-request abstraction, which arbitrarily splits each request at any token boundary into at most two cooperating segments. A two-level scheduling framework then balances micro-request load across unified GPU instances. The global scheduler rapidly selects per-request split points by considering both the request's prefill/decode time ratio and the current load across GPU instances. The local schedulers on each GPU instance independently form SLO-aware batches, adjusting their composition in response to workload fluctuations, potential latency spikes and per-GPU under/over utilization. On real-world traces, DynaServe boosts the overall serving capacity from 1.15$\times$ to 3.07$\times$, improves goodput by up to 1.91$\times$ and 1.61$\times$, and improves the performance by up to 60\% in a hybrid workload under SLO compared to state-of-the-art colocated and disaggregated baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16170v1">When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) admit their mistakes when they should know better? In this work, we define the behavior of acknowledging errors in previously generated answers as "retraction" and aim to understand when and why LLMs choose to retract. We first construct model-specific datasets to evaluate whether a model will retract an incorrect answer that contradicts its own parametric knowledge. While LLMs are capable of retraction, they do so only infrequently. We demonstrate that retraction is closely tied to previously identified indicators of models' internal belief: models fail to retract wrong answers that they "believe" to be factually correct. Steering experiments further demonstrate that internal belief causally influences model retraction. In particular, when the model does not believe its answer, this not only encourages the model to attempt to verify the answer, but also alters attention behavior during self-verification. Finally, we demonstrate that simple supervised fine-tuning significantly improves retraction performance by helping the model learn more accurate internal beliefs. Code and datasets are available on https://github.com/ayyyq/llm-retraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16164v1">Can LLMs Simulate Human Behavioral Variability? A Case Study in the Phonemic Fluency Task</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly explored as substitutes for human participants in cognitive tasks, but their ability to simulate human behavioral variability remains unclear. This study examines whether LLMs can approximate individual differences in the phonemic fluency task, where participants generate words beginning with a target letter. We evaluated 34 model configurations, varying prompt specificity, sampling temperature, and model type, and compared outputs to responses from 106 human participants. While some configurations, especially Claude 3.7 Sonnet, matched human averages and lexical preferences, none reproduced the scope of human variability. LLM outputs were consistently less diverse and structurally rigid, and LLM ensembles failed to increase diversity. Network analyses further revealed fundamental differences in retrieval structure between humans and models. These results highlight key limitations in using LLMs to simulate human cognition and behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16142v1">Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Distilling reasoning paths from teacher to student models via supervised fine-tuning (SFT) provides a shortcut for improving the reasoning ability of smaller Large Language Models (LLMs). However, the reasoning paths generated by teacher models often reflect only surface-level traces of their underlying authentic reasoning. Insights from cognitive neuroscience suggest that authentic reasoning involves a complex interweaving between meta-reasoning (which selects appropriate sub-problems from multiple candidates) and solving (which addresses the sub-problem). This implies authentic reasoning has an implicit multi-branch structure. Supervised fine-tuning collapses this rich structure into a flat sequence of token prediction in the teacher's reasoning path, preventing effective distillation of this structure to students. To address this limitation, we propose RLKD, a reinforcement learning (RL)-based distillation framework guided by a novel Generative Structure Reward Model (GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving steps and computes rewards to measure structural alignment between student and teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to internalize the teacher's implicit multi-branch reasoning structure rather than merely mimicking fixed output paths. Experiments show RLKD surpasses standard SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime, unlocking greater student reasoning potential than SFT-based distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16129v1">LLMs Are Not Scorers: Rethinking MT Evaluation with Generation-Based Methods</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 5 pages, 2 figures, 2 tables. Conforms to the ACL Rolling Review (ARR) short paper track. Code and data available at: https://github.com/CuiNiki/LLMs-Are-Not-Scorers
    </div>
    <details class="paper-abstract">
      Recent studies have applied large language models (LLMs) to machine translation quality estimation (MTQE) by prompting models to assign numeric scores. Nonetheless, these direct scoring methods tend to show low segment-level correlation with human judgments. In this paper, we propose a generation-based evaluation paradigm that leverages decoder-only LLMs to produce high-quality references, followed by semantic similarity scoring using sentence embeddings. We conduct the most extensive evaluation to date in MTQE, covering 8 LLMs and 8 language pairs. Empirical results show that our method outperforms both intra-LLM direct scoring baselines and external non-LLM reference-free metrics from MTME. These findings demonstrate the strength of generation-based evaluation and support a shift toward hybrid approaches that combine fluent generation with accurate semantic assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16128v1">Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted to ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Despite LLMs' explicit alignment against demographic stereotypes, they have been shown to exhibit biases under various social contexts. In this work, we find that LLMs exhibit concerning biases in how they associate solution veracity with demographics. Through experiments across five human value-aligned LLMs on mathematics, coding, commonsense, and writing problems, we reveal two forms of such veracity biases: Attribution Bias, where models disproportionately attribute correct solutions to certain demographic groups, and Evaluation Bias, where models' assessment of identical solutions varies based on perceived demographic authorship. Our results show pervasive biases: LLMs consistently attribute fewer correct solutions and more incorrect ones to African-American groups in math and coding, while Asian authorships are least preferred in writing evaluation. In additional studies, we show LLMs automatically assign racially stereotypical colors to demographic groups in visualization code, suggesting these biases are deeply embedded in models' reasoning processes. Our findings indicate that demographic bias extends beyond surface-level stereotypes and social context provocations, raising concerns about LLMs' deployment in educational and evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16120v1">LLM-Powered AI Agent Systems and Their Applications in Industry</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 This is the author's accepted version of the paper accepted to appear at IEEE AIIoT 2025. The final version will be available via IEEE Xplore. \c{opyright}2025 IEEE. Personal use of this material is permitted
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) has reshaped agent systems. Unlike traditional rule-based agents with limited task scope, LLM-powered agents offer greater flexibility, cross-domain reasoning, and natural language interaction. Moreover, with the integration of multi-modal LLMs, current agent systems are highly capable of processing diverse data modalities, including text, images, audio, and structured tabular data, enabling richer and more adaptive real-world behavior. This paper comprehensively examines the evolution of agent systems from the pre-LLM era to current LLM-powered architectures. We categorize agent systems into software-based, physical, and adaptive hybrid systems, highlighting applications across customer service, software development, manufacturing automation, personalized education, financial trading, and healthcare. We further discuss the primary challenges posed by LLM-powered agents, including high inference latency, output uncertainty, lack of evaluation metrics, and security vulnerabilities, and propose potential solutions to mitigate these concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16118v1">Semiotic Reconstruction of Destination Expectation Constructs An LLM-Driven Computational Paradigm for Social Media Tourism Analytics</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 33 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Social media's rise establishes user-generated content (UGC) as pivotal for travel decisions, yet analytical methods lack scalability. This study introduces a dual-method LLM framework: unsupervised expectation extraction from UGC paired with survey-informed supervised fine-tuning. Findings reveal leisure/social expectations drive engagement more than foundational natural/emotional factors. By establishing LLMs as precision tools for expectation quantification, we advance tourism analytics methodology and propose targeted strategies for experience personalization and social travel promotion. The framework's adaptability extends to consumer behavior research, demonstrating computational social science's transformative potential in marketing optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16113v1">Tools in the Loop: Quantifying Uncertainty of LLM Question Answering Systems That Use Tools</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 10 pages 3 figures 3 tables
    </div>
    <details class="paper-abstract">
      Modern Large Language Models (LLMs) often require external tools, such as machine learning classifiers or knowledge retrieval systems, to provide accurate answers in domains where their pre-trained knowledge is insufficient. This integration of LLMs with external tools expands their utility but also introduces a critical challenge: determining the trustworthiness of responses generated by the combined system. In high-stakes applications, such as medical decision-making, it is essential to assess the uncertainty of both the LLM's generated text and the tool's output to ensure the reliability of the final response. However, existing uncertainty quantification methods do not account for the tool-calling scenario, where both the LLM and external tool contribute to the overall system's uncertainty. In this work, we present a novel framework for modeling tool-calling LLMs that quantifies uncertainty by jointly considering the predictive uncertainty of the LLM and the external tool. We extend previous methods for uncertainty quantification over token sequences to this setting and propose efficient approximations that make uncertainty computation practical for real-world applications. We evaluate our framework on two new synthetic QA datasets, derived from well-known machine learning datasets, which require tool-calling for accurate answers. Additionally, we apply our method to retrieval-augmented generation (RAG) systems and conduct a proof-of-concept experiment demonstrating the effectiveness of our uncertainty metrics in scenarios where external information retrieval is needed. Our results show that the framework is effective in enhancing trust in LLM-based systems, especially in cases where the LLM's internal knowledge is insufficient and external tools are required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01917v2">Steer LLM Latents for Hallucination Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Hallucinations in LLMs pose a significant concern to their safe deployment in real-world applications. Recent approaches have leveraged the latent space of LLMs for hallucination detection, but their embeddings, optimized for linguistic coherence rather than factual accuracy, often fail to clearly separate truthful and hallucinated content. To this end, we propose the Truthfulness Separator Vector (TSV), a lightweight and flexible steering vector that reshapes the LLM's representation space during inference to enhance the separation between truthful and hallucinated outputs, without altering model parameters. Our two-stage framework first trains TSV on a small set of labeled exemplars to form compact and well-separated clusters. It then augments the exemplar set with unlabeled LLM generations, employing an optimal transport-based algorithm for pseudo-labeling combined with a confidence-based filtering process. Extensive experiments demonstrate that TSV achieves state-of-the-art performance with minimal labeled data, exhibiting strong generalization across datasets and providing a practical solution for real-world LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12459v2">LITA: An Efficient LLM-assisted Iterative Topic Augmentation Framework</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Accepted to PAKDD 2025
    </div>
    <details class="paper-abstract">
      Topic modeling is widely used for uncovering thematic structures within text corpora, yet traditional models often struggle with specificity and coherence in domain-focused applications. Guided approaches, such as SeededLDA and CorEx, incorporate user-provided seed words to improve relevance but remain labor-intensive and static. Large language models (LLMs) offer potential for dynamic topic refinement and discovery, yet their application often incurs high API costs. To address these challenges, we propose the LLM-assisted Iterative Topic Augmentation framework (LITA), an LLM-assisted approach that integrates user-provided seeds with embedding-based clustering and iterative refinement. LITA identifies a small number of ambiguous documents and employs an LLM to reassign them to existing or new topics, minimizing API costs while enhancing topic quality. Experiments on two datasets across topic quality and clustering performance metrics demonstrate that LITA outperforms five baseline models, including LDA, SeededLDA, CorEx, BERTopic, and PromptTopic. Our work offers an efficient and adaptable framework for advancing topic modeling and text clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16090v1">Can AI Read Between The Lines? Benchmarking LLMs On Financial Nuance</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 6 pages, 4 figures. Research conducted as part of a Microsoft-sponsored Capstone Project at Santa Clara University
    </div>
    <details class="paper-abstract">
      As of 2025, Generative Artificial Intelligence (GenAI) has become a central tool for productivity across industries. Beyond text generation, GenAI now plays a critical role in coding, data analysis, and research workflows. As large language models (LLMs) continue to evolve, it is essential to assess the reliability and accuracy of their outputs, especially in specialized, high-stakes domains like finance. Most modern LLMs transform text into numerical vectors, which are used in operations such as cosine similarity searches to generate responses. However, this abstraction process can lead to misinterpretation of emotional tone, particularly in nuanced financial contexts. While LLMs generally excel at identifying sentiment in everyday language, these models often struggle with the nuanced, strategically ambiguous language found in earnings call transcripts. Financial disclosures frequently embed sentiment in hedged statements, forward-looking language, and industry-specific jargon, making it difficult even for human analysts to interpret consistently, let alone AI models. This paper presents findings from the Santa Clara Microsoft Practicum Project, led by Professor Charlie Goldenberg, which benchmarks the performance of Microsoft's Copilot, OpenAI's ChatGPT, Google's Gemini, and traditional machine learning models for sentiment analysis of financial text. Using Microsoft earnings call transcripts, the analysis assesses how well LLM-derived sentiment correlates with market sentiment and stock movements and evaluates the accuracy of model outputs. Prompt engineering techniques are also examined to improve sentiment analysis results. Visualizations of sentiment consistency are developed to evaluate alignment between tone and stock performance, with sentiment trends analyzed across Microsoft's lines of business to determine which segments exert the greatest influence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16086v1">Optimizing LLM-Based Multi-Agent System with Textual Feedback: A Case Study on Software Development</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      We have seen remarkable progress in large language models (LLMs) empowered multi-agent systems solving complex tasks necessitating cooperation among experts with diverse skills. However, optimizing LLM-based multi-agent systems remains challenging. In this work, we perform an empirical case study on group optimization of role-based multi-agent systems utilizing natural language feedback for challenging software development tasks under various evaluation dimensions. We propose a two-step agent prompts optimization pipeline: identifying underperforming agents with their failure explanations utilizing textual feedback and then optimizing system prompts of identified agents utilizing failure explanations. We then study the impact of various optimization settings on system performance with two comparison groups: online against offline optimization and individual against group optimization. For group optimization, we study two prompting strategies: one-pass and multi-pass prompting optimizations. Overall, we demonstrate the effectiveness of our optimization method for role-based multi-agent systems tackling software development tasks evaluated on diverse evaluation dimensions, and we investigate the impact of diverse optimization settings on group behaviors of the multi-agent systems to provide practical insights for future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13247v2">An In-Depth Investigation of Data Collection in LLM App Ecosystems</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 Accepted by the ACM Internet Measurement Conference (IMC) 2025
    </div>
    <details class="paper-abstract">
      LLM app (tool) ecosystems are rapidly evolving to support sophisticated use cases that often require extensive user data collection. Given that LLM apps are developed by third parties and anecdotal evidence indicating inconsistent enforcement of policies by LLM platforms, sharing user data with these apps presents significant privacy risks. In this paper, we aim to bring transparency in data practices of LLM app ecosystems. We examine OpenAI's GPT app ecosystem as a case study. We propose an LLM-based framework to analyze the natural language specifications of GPT Actions (custom tools) and assess their data collection practices. Our analysis reveals that Actions collect excessive data across 24 categories and 145 data types, with third-party Actions collecting 6.03% more data on average. We find that several Actions violate OpenAI's policies by collecting sensitive information, such as passwords, which is explicitly prohibited by OpenAI. Lastly, we develop an LLM-based privacy policy analysis framework to automatically check the consistency of data collection by Actions with disclosures in their privacy policies. Our measurements indicate that the disclosures for most of the collected data types are omitted, with only 5.8% of Actions clearly disclosing their data collection practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14652v2">General-Reasoner: Advancing LLM Reasoning Across All Domains</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15793v1">HCRMP: A LLM-Hinted Contextual Reinforcement Learning Framework for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) with Reinforcement Learning (RL) can enhance autonomous driving (AD) performance in complex scenarios. However, current LLM-Dominated RL methods over-rely on LLM outputs, which are prone to hallucinations.Evaluations show that state-of-the-art LLM indicates a non-hallucination rate of only approximately 57.95% when assessed on essential driving-related tasks. Thus, in these methods, hallucinations from the LLM can directly jeopardize the performance of driving policies. This paper argues that maintaining relative independence between the LLM and the RL is vital for solving the hallucinations problem. Consequently, this paper is devoted to propose a novel LLM-Hinted RL paradigm. The LLM is used to generate semantic hints for state augmentation and policy optimization to assist RL agent in motion planning, while the RL agent counteracts potential erroneous semantic indications through policy learning to achieve excellent driving performance. Based on this paradigm, we propose the HCRMP (LLM-Hinted Contextual Reinforcement Learning Motion Planner) architecture, which is designed that includes Augmented Semantic Representation Module to extend state space. Contextual Stability Anchor Module enhances the reliability of multi-critic weight hints by utilizing information from the knowledge base. Semantic Cache Module is employed to seamlessly integrate LLM low-frequency guidance with RL high-frequency control. Extensive experiments in CARLA validate HCRMP's strong overall driving performance. HCRMP achieves a task success rate of up to 80.3% under diverse driving conditions with different traffic densities. Under safety-critical driving conditions, HCRMP significantly reduces the collision rate by 11.4%, which effectively improves the driving performance in complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24245v2">Enhancing Large Language Models (LLMs) for Telecommunications using Knowledge Graphs and Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 This work has been accepted to ICC 2025 IEEE International Conference on Communications. copyright 2025 IEEE
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant progress in general-purpose natural language processing tasks. However, LLMs are still facing challenges when applied to domain-specific areas like telecommunications, which demands specialized expertise and adaptability to evolving standards. This paper presents a novel framework that combines knowledge graph (KG) and retrieval-augmented generation (RAG) techniques to enhance LLM performance in the telecom domain. The framework leverages a KG to capture structured, domain-specific information about network protocols, standards, and other telecom-related entities, comprehensively representing their relationships. By integrating KG with RAG, LLMs can dynamically access and utilize the most relevant and up-to-date knowledge during response generation. This hybrid approach bridges the gap between structured knowledge representation and the generative capabilities of LLMs, significantly enhancing accuracy, adaptability, and domain-specific comprehension. Our results demonstrate the effectiveness of the KG-RAG framework in addressing complex technical queries with precision. The proposed KG-RAG model attained an accuracy of 88% for question answering tasks on a frequently used telecom-specific dataset, compared to 82% for the RAG-only and 48% for the LLM-only approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15740v1">HybridProver: Augmenting Theorem Proving with LLM-Driven Proof Synthesis and Refinement</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Formal methods is pivotal for verifying the reliability of critical systems through rigorous mathematical proofs. However, its adoption is hindered by labor-intensive manual proofs and the expertise required to use theorem provers. Recent advancements in large language models (LLMs) offer new opportunities for automated theorem proving. Two promising approaches are generating tactics step by step and generating a whole proof directly with an LLM. However, existing work makes no attempt to combine the two approaches. In this work, we introduce HybridProver, a dual-model proof synthesis framework that combines tactic-based generation and whole-proof synthesis to harness the benefits of both approaches. HybridProver generates whole proof candidates for evaluation directly, then extracts proof sketches from those candidates. It then uses a tactic-based generation model that integrates automated tools to complete the sketches via stepwise refinement. We implement HybridProver for the Isabelle theorem prover and fine-tune LLMs on our optimized Isabelle datasets. Evaluation on the miniF2F dataset illustrates HybridProver's effectiveness. We achieve a 59.4% success rate on miniF2F, where the previous SOTA is 56.1%. Our ablation studies show that this SOTA result is attributable to combining whole-proof and tactic-based generation. Additionally, we show how the dataset quality, training parameters, and sampling diversity affect the final result during automated theorem proving with LLMs. All of our code, datasets, and LLMs are open source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15738v1">Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15710v1">Advancing LLM Safe Alignment with Safety Representation Ranking</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has demonstrated milestone success in a variety of tasks, yet their potential for generating harmful content has raised significant safety concerns. Existing safety evaluation approaches typically operate directly on textual responses, overlooking the rich information embedded in the model's internal representations. In this paper, we propose Safety Representation Ranking (SRR), a listwise ranking framework that selects safe responses using hidden states from the LLM itself. SRR encodes both instructions and candidate completions using intermediate transformer representations and ranks candidates via a lightweight similarity-based scorer. Our approach directly leverages internal model states and supervision at the list level to capture subtle safety signals. Experiments across multiple benchmarks show that SRR significantly improves robustness to adversarial prompts. Our code will be available upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15683v1">A Federated Splitting Framework for LLMs: Security, Efficiency, and Adaptability</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Private data is typically larger and of higher quality than public data, offering great potential to improve LLM. However, its scattered distribution across data silos and the high computational demands of LLMs limit their deployment in federated environments. To address this, the transformer-based split learning model has emerged, offloading most model parameters to the server while retaining only the embedding and output layers on clients to ensure privacy. However, it still faces significant challenges in security, efficiency, and adaptability: 1) embedding gradients are vulnerable to attacks, leading to reverse engineering of private data; 2) the autoregressive nature of LLMs means that federated split learning can only train and infer sequentially, causing high communication overhead; 3) fixed partition points lack adaptability to downstream tasks. In this paper, we introduce FL-LLaMA, a secure, efficient, and adaptive federated split framework based on LLaMA2. First, we place some input and output blocks on the local client and inject Gaussian noise into forward-pass hidden states, enabling secure end-to-end propagation. Second, we employ client-batch and server-hierarchical strategies to achieve parallel training, along with attention-mask compression and KV cache mechanisms to accelerate inference, reducing communication costs effectively. Third, we allow users to dynamically adjust the partition points for input/output blocks based on specific task requirements and hardware limitations. Experiments on NLU, summarization and conversational QA tasks show that FL-LLaMA maintains performance comparable to centralized LLaMA2, and achieves up to 2x train speedups and 8x inference speedups. Further analysis of privacy attacks and different partition points also demonstrates the effectiveness of FL-LLaMA in security and adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17216v2">Intermediate Languages Matter: Formal Choice Drives Neurosymbolic LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve astonishing results on a wide range of tasks. However, their formal reasoning ability still lags behind. A promising approach is Neurosymbolic LLM reasoning. It works by using LLMs as translators from natural to formal languages and symbolic solvers for deriving correct results. Still, it remains unclear what the contributing factors to the success of Neurosymbolic LLM reasoning are. This paper shows that one important factor is the choice of the formal language. By comparing 4 formal languages on 3 datasets over 6 LLMs, we show that the choice of formal language affects both the syntactic and the semantic reasoning capability. Thereby, we introduce the intermediate language challenge, which is the challenge of picking a suitable formal language for neurosymbolic reasoning. Further, we compare the effects of using different in-context-learning examples in an ablation study. We conclude that on average, context-aware encodings help LLMs to reason, while there is no apparent effect of using comments or markdown syntax.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10900v2">Explain What You Mean: Intent Augmented Knowledge Graph Recommender Built With An LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Interaction sparsity is a long-standing challenge in recommendation systems. Sparsity manifests in environments with disproportional cardinality of groupings of entities, such as users and products in an online marketplace. It is also found for newly introduced entities, described as the cold-start problem. Recent efforts to mitigate this issue either enrich the connectivity data by incorporating social networks or external knowledge graphs, or fine-tune LLMs into interaction augmenters or next-item recommenders. However, these techniques tend to be resource demanding, requiring high computational power. They also have several limitations, including data availability, low quality, or synthetic noise issues. In this work, we propose LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that leverages retrieval-augmented generation and an encoding approach to construct and densify a knowledge graph. IKGR leverages latent user-item affinities from an interaction knowledge graph and further densifies it through mutual intent connectivity. This addresses sparsity issues and allows the model to make intent-grounded recommendations with an interpretable embedding translation layer. Through extensive experiments on real-world datasets, we demonstrate that IKGR overcomes knowledge gaps and achieves substantial gains over state-of-the-art baselines on both publicly available and our internal recommendation datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15656v1">Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Fine-tuning on open-source Large Language Models (LLMs) with proprietary data is now a standard practice for downstream developers to obtain task-specific LLMs. Surprisingly, we reveal a new and concerning risk along with the practice: the creator of the open-source LLMs can later extract the private downstream fine-tuning data through simple backdoor training, only requiring black-box access to the fine-tuned downstream model. Our comprehensive experiments, across 4 popularly used open-source models with 3B to 32B parameters and 2 downstream datasets, suggest that the extraction performance can be strikingly high: in practical settings, as much as 76.3% downstream fine-tuning data (queries) out of a total 5,000 samples can be perfectly extracted, and the success rate can increase to 94.9% in more ideal settings. We also explore a detection-based defense strategy but find it can be bypassed with improved attack. Overall, we highlight the emergency of this newly identified data breaching risk in fine-tuning, and we hope that more follow-up research could push the progress of addressing this concerning risk. The code and data used in our experiments are released at https://github.com/thu-coai/Backdoor-Data-Extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15623v1">Can LLMs $\textit{understand}$ Math? -- Exploring the Pitfalls in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate considerable potential in various natural language tasks but face significant challenges in mathematical reasoning, particularly in executing precise, multi-step logic. However, current evaluation frameworks judge their performance solely based on accuracy, which only accounts for the final answer. This study explores these pitfalls by employing a novel evaluation framework. We propose an evaluation metric called the MAPLE score, which holistically quantifies reasoning misalignment by integrating error rates, redundancy, and validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15607v1">From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 David Dinucu-Jianu and Jakub Macina contributed equally. Code available: https://github.com/eth-lre/PedagogicalRL
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can transform education, but their optimization for direct question-answering often undermines effective pedagogy which requires strategically withholding answers. To mitigate this, we propose an online reinforcement learning (RL)-based alignment framework that can quickly adapt LLMs into effective tutors using simulated student-tutor interactions by emphasizing pedagogical quality and guided problem-solving over simply giving away answers. We use our method to train a 7B parameter tutor model without human annotations which reaches similar performance to larger proprietary models like LearnLM. We introduce a controllable reward weighting to balance pedagogical support and student solving accuracy, allowing us to trace the Pareto frontier between these two objectives. Our models better preserve reasoning capabilities than single-turn SFT baselines and can optionally enhance interpretability through thinking tags that expose the model's instructional planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10259v3">SpecOffload: Unlocking Latent GPU Capacity for LLM Inference on Resource-Constrained Devices</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Efficient LLM inference on resource-constrained devices presents significant challenges in compute and memory utilization. Due to limited GPU memory, existing systems offload model weights to CPU memory, incurring substantial I/O overhead between the CPU and GPU. This leads to two major inefficiencies: (1) GPU cores are underutilized, often remaining idle while waiting for data to be loaded; and (2) GPU memory has low impact on performance, as reducing its capacity has minimal effect on overall throughput.In this paper, we propose SpecOffload, a high-throughput inference engine that embeds speculative decoding into offloading. Our key idea is to unlock latent GPU resources for storing and executing a draft model used for speculative decoding, thus accelerating inference at near-zero additional cost. To support this, we carefully orchestrate the interleaved execution of target and draft models in speculative decoding within the offloading pipeline, and propose a planner to manage tensor placement and select optimal parameters. Compared to the best baseline, SpecOffload improves GPU core utilization by 4.49x and boosts inference throughput by 2.54x. Our code is available at https://github.com/MobiSense/SpecOffload-public .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15596v1">Exploring LLM-Generated Feedback for Economics Essays: How Teaching Assistants Evaluate and Envision Its Use</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 To be published in AIED'2025: In Proceedings of the 26th International Conference on Artificial Intelligence in Education. The system prompt and example feedback can be found through http://github.com/UM-Lifelong-Learning-Lab/AIED2025-Exploring-LLM-Generated-Feedback-for-Economics-Essay
    </div>
    <details class="paper-abstract">
      This project examines the prospect of using AI-generated feedback as suggestions to expedite and enhance human instructors' feedback provision. In particular, we focus on understanding the teaching assistants' perspectives on the quality of AI-generated feedback and how they may or may not utilize AI feedback in their own workflows. We situate our work in a foundational college Economics class, which has frequent short essay assignments. We developed an LLM-powered feedback engine that generates feedback on students' essays based on grading rubrics used by the teaching assistants (TAs). To ensure that TAs can meaningfully critique and engage with the AI feedback, we had them complete their regular grading jobs. For a randomly selected set of essays that they had graded, we used our feedback engine to generate feedback and displayed the feedback as in-text comments in a Word document. We then performed think-aloud studies with 5 TAs over 20 1-hour sessions to have them evaluate the AI feedback, contrast the AI feedback with their handwritten feedback, and share how they envision using the AI feedback if they were offered as suggestions. The study highlights the importance of providing detailed rubrics for AI to generate high-quality feedback for knowledge-intensive essays. TAs considered that using AI feedback as suggestions during their grading could expedite grading, enhance consistency, and improve overall feedback quality. We discuss the importance of decomposing the feedback generation task into steps and presenting intermediate results, in order for TAs to use the AI feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14336v2">Scaling and Enhancing LLM-based AVSR: A Sparse Mixture of Projectors Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 Interspeech 2025
    </div>
    <details class="paper-abstract">
      Audio-Visual Speech Recognition (AVSR) enhances robustness in noisy environments by integrating visual cues. While recent advances integrate Large Language Models (LLMs) into AVSR, their high computational cost hinders deployment in resource-constrained settings. To address this, we propose Llama-SMoP, an efficient Multimodal LLM that employs a Sparse Mixture of Projectors (SMoP) module to scale model capacity without increasing inference costs. By incorporating sparsely-gated mixture-of-experts (MoE) projectors, Llama-SMoP enables the use of smaller LLMs while maintaining strong performance. We explore three SMoP configurations and show that Llama-SMoP DEDR (Disjoint-Experts, Disjoint-Routers), which uses modality-specific routers and experts, achieves superior performance on ASR, VSR, and AVSR tasks. Ablation studies confirm its effectiveness in expert activation, scalability, and noise robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13178v4">Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 17 pages, 3 fugures
    </div>
    <details class="paper-abstract">
      Post-training Quantization (PTQ) technique has been extensively adopted for large language models (LLMs) compression owing to its efficiency and low resource requirement. However, current research lacks a in-depth analysis of the superior and applicable scenarios of each PTQ strategy. In addition, existing algorithms focus primarily on performance, overlooking the trade-off among model size, performance, and quantization bitwidth. To mitigate these confusions, we provide a novel benchmark for LLMs PTQ in this paper. Firstly, in order to support our benchmark, we propose a comprehensive taxonomy for existing mainstream methods by scrutinizing their computational strategies (e.g., optimization-based, compensation-based, etc.). Then, we conduct extensive experiments with the baseline within each class, covering models with various sizes (7B-70B), bitwidths, training levels (LLaMA1/2/3/3.1), architectures (Mixtral, DeepSeekMoE and Mamba) and modality (LLaVA1.5 and VILA1.5) on a wide range of evaluation metrics.Through comparative analysis on the results, we summarize the superior of each PTQ strategy and modelsize-bitwidth trade-off considering the performance. For example, our benchmark reveals that compensation-based technique demonstrates outstanding cross-architecture robustness and extremely low-bit PTQ for ultra large models should be reexamined. Finally, we further accordingly claim that a practical combination of compensation and other PTQ strategy can achieve SOTA various robustness. We believe that our benchmark will provide valuable recommendations for the deployment of LLMs and future research on PTQ approaches.We conduct an repository for our benchmark at https://github.com/zjq0455/PTQ_Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17644v4">BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Serving systems for Large Language Models (LLMs) are often optimized to improve quality of service (QoS) and throughput. However, due to the lack of open-source LLM serving workloads, these systems are frequently evaluated under unrealistic workload assumptions. Consequently, performance may degrade when systems are deployed in real-world scenarios. This work presents BurstGPT, an LLM serving workload with 10.31 million traces from regional Azure OpenAI GPT services over 213 days. BurstGPT captures LLM serving characteristics from user, model and system perspectives: (1) User request concurrency: burstiness variations of requests in Azure OpenAI GPT services, revealing diversified concurrency patterns in different services and model types. (2) User conversation patterns: counts and intervals within conversations for service optimizations. (3) Model response lengths: auto-regressive serving processes of GPT models, showing statistical relations between requests and their responses. (4) System response failures: failures of conversation and API services, showing intensive resource needs and limited availability of LLM services in Azure. The details of the characteristics can serve multiple purposes in LLM serving optimizations, such as system evaluation and trace provisioning. In our demo evaluation with BurstGPT, frequent variations in BurstGPT reveal declines in efficiency, stability, or reliability in realistic LLM serving. We identify that the generalization of KV cache management, scheduling and disaggregation optimizations can be improved under realistic workload evaluations. BurstGPT is publicly available now at https://github.com/HPMLL/BurstGPT and is widely used to develop prototypes of LLM serving frameworks in the industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15524v1">Evaluate Bias without Manual Test Sets: A Concept Representation Perspective for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Bias in Large Language Models (LLMs) significantly undermines their reliability and fairness. We focus on a common form of bias: when two reference concepts in the model's concept space, such as sentiment polarities (e.g., "positive" and "negative"), are asymmetrically correlated with a third, target concept, such as a reviewing aspect, the model exhibits unintended bias. For instance, the understanding of "food" should not skew toward any particular sentiment. Existing bias evaluation methods assess behavioral differences of LLMs by constructing labeled data for different social groups and measuring model responses across them, a process that requires substantial human effort and captures only a limited set of social concepts. To overcome these limitations, we propose BiasLens, a test-set-free bias analysis framework based on the structure of the model's vector space. BiasLens combines Concept Activation Vectors (CAVs) with Sparse Autoencoders (SAEs) to extract interpretable concept representations, and quantifies bias by measuring the variation in representational similarity between the target concept and each of the reference concepts. Even without labeled data, BiasLens shows strong agreement with traditional bias evaluation metrics (Spearman correlation r > 0.85). Moreover, BiasLens reveals forms of bias that are difficult to detect using existing methods. For example, in simulated clinical scenarios, a patient's insurance status can cause the LLM to produce biased diagnostic assessments. Overall, BiasLens offers a scalable, interpretable, and efficient paradigm for bias discovery, paving the way for improving fairness and transparency in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15501v1">Protoknowledge Shapes Behaviour of LLMs in Downstream Tasks: Memorization and Generalization with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      We introduce the concept of protoknowledge to formalize and measure how sequences of tokens encoding Knowledge Graphs are internalized during pretraining and utilized at inference time by Large Language Models (LLMs). Indeed, LLMs have demonstrated the ability to memorize vast amounts of token sequences during pretraining, and a central open question is how they leverage this memorization as reusable knowledge through generalization. We then categorize protoknowledge into lexical, hierarchical, and topological forms, varying on the type of knowledge that needs to be activated. We measure protoknowledge through Knowledge Activation Tasks (KATs), analyzing its general properties such as semantic bias. We then investigate the impact of protoknowledge on Text-to-SPARQL performance by varying prompting strategies depending on input conditions. To this end, we adopt a novel analysis framework that assesses whether model predictions align with the successful activation of the relevant protoknowledge for each query. This methodology provides a practical tool to explore Semantic-Level Data Contamination and serves as an effective strategy for Closed-Pretraining models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15480v1">KaFT: Knowledge-aware Fine-tuning for Boosting LLMs' Domain-specific Question-Answering Performance</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 Accepted to ACL2025 Findings
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a common approach to improve the domain-specific question-answering (QA) performance of large language models (LLMs). However, recent literature reveals that due to the conflicts between LLMs' internal knowledge and the context knowledge of training data, vanilla SFT using the full QA training set is usually suboptimal. In this paper, we first design a query diversification strategy for robust conflict detection and then conduct a series of experiments to analyze the impact of knowledge conflict. We find that 1) training samples with varied conflicts contribute differently, where SFT on the data with large conflicts leads to catastrophic performance drops; 2) compared to directly filtering out the conflict data, appropriately applying the conflict data would be more beneficial. Motivated by this, we propose a simple-yet-effective Knowledge-aware Fine-tuning (namely KaFT) approach to effectively boost LLMs' performance. The core of KaFT is to adapt the training weight by assigning different rewards for different training samples according to conflict level. Extensive experiments show that KaFT brings consistent and significant improvements across four LLMs. More analyses prove that KaFT effectively improves the model generalization and alleviates the hallucination.
    </details>
</div>
