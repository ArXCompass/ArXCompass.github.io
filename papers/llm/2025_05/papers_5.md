# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09501v3">ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Recent research on Reasoning of Large Language Models (LLMs) has sought to further enhance their performance by integrating meta-thinking -- enabling models to monitor, evaluate, and control their reasoning processes for more adaptive and effective problem-solving. However, current single-agent work lacks a specialized design for acquiring meta-thinking, resulting in low efficacy. To address this challenge, we introduce Reinforced Meta-thinking Agents (ReMA), a novel framework that leverages Multi-Agent Reinforcement Learning (MARL) to elicit meta-thinking behaviors, encouraging LLMs to think about thinking. ReMA decouples the reasoning process into two hierarchical agents: a high-level meta-thinking agent responsible for generating strategic oversight and plans, and a low-level reasoning agent for detailed executions. Through iterative reinforcement learning with aligned objectives, these agents explore and learn collaboration, leading to improved generalization and robustness. Empirical results from single-turn experiments demonstrate that ReMA outperforms single-agent RL baselines on complex reasoning tasks, including competitive-level mathematical benchmarks and LLM-as-a-Judge benchmarks. Additionally, we further extend ReMA to multi-turn interaction settings, leveraging turn-level ratio and parameter sharing to improve efficiency. Comprehensive ablation studies further illustrate the evolving dynamics of each distinct agent, providing valuable insights into how the meta-thinking reasoning process enhances the reasoning capabilities of LLMs. Our code can be found in https://github.com/ziyuwan/ReMA-public
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14838v4">DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Efficient KV cache management in LLMs is crucial for long-context tasks like RAG and summarization. Existing KV cache compression methods enforce a fixed pattern, neglecting task-specific characteristics and reducing the retention of essential information. However, we observe distinct activation patterns across layers in various tasks, highlighting the need for adaptive strategies tailored to each task's unique demands. Based on this insight, we propose DynamicKV, a method that dynamically optimizes token retention by adjusting the number of tokens retained at each layer to adapt to the specific task. DynamicKV establishes global and per-layer maximum KV cache budgets, temporarily retaining the maximum budget for the current layer, and periodically updating the KV cache sizes of all preceding layers during inference. Our method retains only 1.7% of the KV cache size while achieving ~85% of the Full KV cache performance on LongBench. Notably, even under extreme compression (0.9%), DynamicKV surpasses state-of-the-art (SOTA) methods by 11% in the Needle-in-a-Haystack test using Mistral-7B-Instruct-v0.2. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20658v1">Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 13 pages, 5 figures, published to ACL
    </div>
    <details class="paper-abstract">
      Temporal Logic (TL), especially Signal Temporal Logic (STL), enables precise formal specification, making it widely used in cyber-physical systems such as autonomous driving and robotics. Automatically transforming NL into STL is an attractive approach to overcome the limitations of manual transformation, which is time-consuming and error-prone. However, due to the lack of datasets, automatic transformation currently faces significant challenges and has not been fully explored. In this paper, we propose an NL-STL dataset named STL-Diversity-Enhanced (STL-DivEn), which comprises 16,000 samples enriched with diverse patterns. To develop the dataset, we first manually create a small-scale seed set of NL-STL pairs. Next, representative examples are identified through clustering and used to guide large language models (LLMs) in generating additional NL-STL pairs. Finally, diversity and accuracy are ensured through rigorous rule-based filters and human validation. Furthermore, we introduce the Knowledge-Guided STL Transformation (KGST) framework, a novel approach for transforming natural language into STL, involving a generate-then-refine process based on external knowledge. Statistical analysis shows that the STL-DivEn dataset exhibits more diversity than the existing NL-STL dataset. Moreover, both metric-based and human evaluations indicate that our KGST approach outperforms baseline models in transformation accuracy on STL-DivEn and DeepSTL datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07266v3">When More is Less: Understanding Chain-of-Thought Length in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) employ Chain-of-Thought (CoT) reasoning to deconstruct complex problems. While longer CoTs are often presumed superior, this paper challenges that notion, arguing that longer is not always better. Drawing on combined evidence from real-world observations, controlled experiments, and theoretical analysis, we demonstrate that task accuracy typically follows an inverted U-shaped curve with CoT length, where performance initially improves but eventually decreases as the number of CoT steps increases. With controlled experiments, we further uncover the scaling behaviors of the optimal CoT length: it increases with task difficulty but decreases with model capability, exposing an inherent simplicity bias where more capable models favor shorter, more efficient CoT reasoning. This bias is also evident in Reinforcement Learning (RL) training, where models gravitate towards shorter CoTs as their accuracy improves. To have a deep understanding of these dynamics, we establish a simple theoretical model that formally proves these phenomena, including the optimal length's scaling laws and the emergence of simplicity bias during RL. Guided by this framework, we demonstrate significant practical benefits from training with optimally-lengthed CoTs and employing length-aware filtering at inference. These findings offer both a principled understanding of the "overthinking" phenomenon and multiple practical guidelines for CoT calibration, enabling LLMs to achieve optimal reasoning performance with adaptive CoTs tailored to task complexity and model capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20650v1">FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      We introduce FinTagging, the first full-scope, table-aware XBRL benchmark designed to evaluate the structured information extraction and semantic alignment capabilities of large language models (LLMs) in the context of XBRL-based financial reporting. Unlike prior benchmarks that oversimplify XBRL tagging as flat multi-class classification and focus solely on narrative text, FinTagging decomposes the XBRL tagging problem into two subtasks: FinNI for financial entity extraction and FinCL for taxonomy-driven concept alignment. It requires models to jointly extract facts and align them with the full 10k+ US-GAAP taxonomy across both unstructured text and structured tables, enabling realistic, fine-grained evaluation. We assess a diverse set of LLMs under zero-shot settings, systematically analyzing their performance on both subtasks and overall tagging accuracy. Our results reveal that, while LLMs demonstrate strong generalization in information extraction, they struggle with fine-grained concept alignment, particularly in disambiguating closely related taxonomy entries. These findings highlight the limitations of existing LLMs in fully automating XBRL tagging and underscore the need for improved semantic reasoning and schema-aware modeling to meet the demands of accurate financial disclosure. Code is available at our GitHub repository and data is at our Hugging Face repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19419v2">It's Not Just Labeling -- A Research on LLM Generated Feedback Interpretability and Image Labeling Sketch Features</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      The quality of training data is critical to the performance of machine learning applications in domains like transportation, healthcare, and robotics. Accurate image labeling, however, often relies on time-consuming, expert-driven methods with limited feedback. This research introduces a sketch-based annotation approach supported by large language models (LLMs) to reduce technical barriers and enhance accessibility. Using a synthetic dataset, we examine how sketch recognition features relate to LLM feedback metrics, aiming to improve the reliability and interpretability of LLM-assisted labeling. We also explore how prompting strategies and sketch variations influence feedback quality. Our main contribution is a sketch-based virtual assistant that simplifies annotation for non-experts and advances LLM-driven labeling tools in terms of scalability, accessibility, and explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04948v2">Prompt-Based LLMs for Position Bias-Aware Reranking in Personalized Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 We have decided to withdraw the manuscript as it requires substantial revisions that go beyond what is appropriate for a versioned update on arXiv. We plan to resubmit once the necessary improvements are made
    </div>
    <details class="paper-abstract">
      Recommender systems are essential for delivering personalized content across digital platforms by modeling user preferences and behaviors. Recently, large language models (LLMs) have been adopted for prompt-based recommendation due to their ability to generate personalized outputs without task-specific training. However, LLM-based methods face limitations such as limited context window size, inefficient pointwise and pairwise prompting, and difficulty handling listwise ranking due to token constraints. LLMs can also be sensitive to position bias, as they may overemphasize earlier items in the prompt regardless of their true relevance. To address and investigate these issues, we propose a hybrid framework that combines a traditional recommendation model with an LLM for reranking top-k items using structured prompts. We evaluate the effects of user history reordering and instructional prompts for mitigating position bias. Experiments on MovieLens-100K show that randomizing user history improves ranking quality, but LLM-based reranking does not outperform the base model. Explicit instructions to reduce position bias are also ineffective. Our evaluations reveal limitations in LLMs' ability to model ranking context and mitigate bias. Our code is publicly available at https://github.com/aminul7506/LLMForReRanking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v8">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20643v1">Can Past Experience Accelerate LLM Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Allocating more compute to large language models (LLMs) reasoning has generally been demonstrated to improve their effectiveness, but also results in increased inference time. In contrast, humans can perform tasks faster and better with increased experience and exposure. Hence, this paper aims to investigate the question: Can LLMs also become faster at reasoning through recurrent exposure on relevant tasks, and if so, how can it be achieved? To address these questions, we first formalize the problem setting of LLM reasoning speedup systematically in the dimensions of task relevancy and compute budget calculation. We then propose SpeedupLLM, a theoretically guaranteed framework to implement and benchmark such reasoning speedup behaviour based on adaptive compute allocation and memory mechanisms. We further conduct comprehensive experiments to benchmark such behaviour across different question similarity levels, memory methods, and reasoning methods. Results show that LLMs can generally reason faster with past experience, achieving up to a 56% reduction in compute cost when equipped with appropriate memory and reasoning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16216v3">GMoE: Empowering LLMs Fine-Tuning via MoE Graph Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 9 pages, 25 figures
    </div>
    <details class="paper-abstract">
      The sparse Mixture-of-Experts (MoE) architecture of large language models (LLMs) confronts an inherent issue of load imbalance arising from the simplistic linear router strategy, which ultimately causes the instability and inefficient learning of LLMs. To address this challenge, we introduce a novel MoE graph-based framework $\textbf{GMoE}$, aimed at enhancing the collaboration among multiple experts. In GMoE, a graph router function is designed to capture the collaboration signals among experts. This enables all experts to dynamically allocate information derived from input data by sharing information with their neighboring experts. Moreover, we put forward two coordination strategies in GMoE: the $\textit{Poisson distribution-based distinction strategy}$ and the $\textit{Normal distribution-based balance strategy}$, to further release the capacity of each expert and increase the model stability in the fine-tuning of LLMs. Specifically, we leverage a parameter-efficient fine-tuning technique, i.e., Low-Rank Adaptation (LoRA), to implement the graph MoE architecture. Extensive experiments on four real-world benchmark datasets demonstrate the effectiveness of GMoE, showing the benefits of facilitating collaborations of multiple experts in LLM fine-tuning. The code of experimental implementation is available at https://github.com/BAI-LAB/GMoE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16522v2">Benchmarking and Pushing the Multi-Bias Elimination Boundary of LLMs via Causal Effect Estimation-guided Debiasing</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Despite significant progress, recent studies have indicated that current large language models (LLMs) may still utilize bias during inference, leading to the poor generalizability of LLMs. Some benchmarks are proposed to investigate the generalizability of LLMs, with each piece of data typically containing one type of controlled bias. However, a single piece of data may contain multiple types of biases in practical applications. To bridge this gap, we propose a multi-bias benchmark where each piece of data contains five types of biases. The evaluations conducted on this benchmark reveal that the performance of existing LLMs and debiasing methods is unsatisfying, highlighting the challenge of eliminating multiple types of biases simultaneously. To overcome this challenge, we propose a causal effect estimation-guided multi-bias elimination method (CMBE). This method first estimates the causal effect of multiple types of biases simultaneously. Subsequently, we eliminate the causal effect of biases from the total causal effect exerted by both the semantic information and biases during inference. Experimental results show that CMBE can effectively eliminate multiple types of bias simultaneously to enhance the generalizability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20627v1">Fundamental Limits of Game-Theoretic LLM Alignment: Smith Consistency and Preference Matching</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Nash Learning from Human Feedback is a game-theoretic framework for aligning large language models (LLMs) with human preferences by modeling learning as a two-player zero-sum game. However, using raw preference as the payoff in the game highly limits the potential of the game-theoretic LLM alignment framework. In this paper, we systematically study using what choices of payoff based on the pairwise human preferences can yield desirable alignment properties. We establish necessary and sufficient conditions for Condorcet consistency, diversity through mixed strategies, and Smith consistency. These results provide a theoretical foundation for the robustness of game-theoretic LLM alignment. Further, we show the impossibility of preference matching -- i.e., no smooth and learnable mappings of pairwise preferences can guarantee a unique Nash equilibrium that matches a target policy, even under standard assumptions like the Bradley-Terry-Luce model. This result highlights the fundamental limitation of game-theoretic LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14434v2">LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Automated feature engineering plays a critical role in improving predictive model performance for tabular learning tasks. Traditional automated feature engineering methods are limited by their reliance on pre-defined transformations within fixed, manually designed search spaces, often neglecting domain knowledge. Recent advances using Large Language Models (LLMs) have enabled the integration of domain knowledge into the feature engineering process. However, existing LLM-based approaches use direct prompting or rely solely on validation scores for feature selection, failing to leverage insights from prior feature discovery experiments or establish meaningful reasoning between feature generation and data-driven performance. To address these challenges, we propose LLM-FE, a novel framework that combines evolutionary search with the domain knowledge and reasoning capabilities of LLMs to automatically discover effective features for tabular learning tasks. LLM-FE formulates feature engineering as a program search problem, where LLMs propose new feature transformation programs iteratively, and data-driven feedback guides the search process. Our results demonstrate that LLM-FE consistently outperforms state-of-the-art baselines, significantly enhancing the performance of tabular prediction models across diverse classification and regression benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02825v5">Randomly Sampled Language Reasoning Problems Explain Limits of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 10 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      While LLMs have revolutionized the field of machine learning due to their high performance across a range of tasks, they are known to perform poorly in planning, hallucinate false answers, have degraded performance on less canonical versions of the same task, and answer incorrectly on a variety of specific prompts. There are several emerging theories of LLM performance with some predictive power, among them that LLMs lack world modeling ability, that they have an undesirable bias towards an autoregressive prior, and that they perform less well on more novel problems. The existing literature on novelty has focused on tasks of relatively high complexity, studying perturbations of canonical but complex problems. In this paper, we attempt to isolate novelty as a factor in LLM underperformance. To this end, we consider an extremely simple domain: next token prediction on simple language tasks. The twist is that these language tasks are unseen, as they are randomly drawn from a large, parsimoniously defined set of languages arising from simple grammar rules. This allows us to isolate the effect of task novelty and see if it is sufficient to explain low performance. We find that LLMs uniformly underperform n-gram models (which do not have the capacity for world modeling) on these tasks, both when used as next token predictors and as reasoners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17049v2">Gender and Positional Biases in LLM-Based Hiring Decisions: Evidence from Comparative CV/Résumé Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      This study examines the behavior of Large Language Models (LLMs) when evaluating professional candidates based on their resumes or curricula vitae (CVs). In an experiment involving 22 leading LLMs, each model was systematically given one job description along with a pair of profession-matched CVs, one bearing a male first name, the other a female first name, and asked to select the more suitable candidate for the job. Each CV pair was presented twice, with names swapped to ensure that any observed preferences in candidate selection stemmed from gendered names cues. Despite identical professional qualifications across genders, all LLMs consistently favored female-named candidates across 70 different professions. Adding an explicit gender field (male/female) to the CVs further increased the preference for female applicants. When gendered names were replaced with gender-neutral identifiers "Candidate A" and "Candidate B", several models displayed a preference to select "Candidate A". Counterbalancing gender assignment between these gender-neutral identifiers resulted in gender parity in candidate selection. When asked to rate CVs in isolation rather than compare pairs, LLMs assigned slightly higher average scores to female CVs overall, but the effect size was negligible. Including preferred pronouns (he/him or she/her) next to a candidate's name slightly increased the odds of the candidate being selected regardless of gender. Finally, most models exhibited a substantial positional bias to select the candidate listed first in the prompt. These findings underscore the need for caution when deploying LLMs in high-stakes autonomous decision-making contexts and raise doubts about whether LLMs consistently apply principled reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21828v1">SAGE-Eval: Evaluating LLMs for Systematic Generalizations of Safety Facts</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Do LLMs robustly generalize critical safety facts to novel situations? Lacking this ability is dangerous when users ask naive questions. For instance, "I'm considering packing melon balls for my 10-month-old's lunch. What other foods would be good to include?" Before offering food options, the LLM should warn that melon balls pose a choking hazard to toddlers, as documented by the CDC. Failing to provide such warnings could result in serious injuries or even death. To evaluate this, we introduce SAGE-Eval, SAfety-fact systematic GEneralization evaluation, the first benchmark that tests whether LLMs properly apply well established safety facts to naive user queries. SAGE-Eval comprises 104 facts manually sourced from reputable organizations, systematically augmented to create 10,428 test scenarios across 7 common domains (e.g., Outdoor Activities, Medicine). We find that the top model, Claude-3.7-sonnet, passes only 58% of all the safety facts tested. We also observe that model capabilities and training compute weakly correlate with performance on SAGE-Eval, implying that scaling up is not the golden solution. Our findings suggest frontier LLMs still lack robust generalization ability. We recommend developers use SAGE-Eval in pre-deployment evaluations to assess model reliability in addressing salient risks. We publicly release SAGE-Eval at https://huggingface.co/datasets/YuehHanChen/SAGE-Eval and our code is available at https://github.com/YuehHanChen/SAGE-Eval/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21815v1">Scientific Paper Retrieval with LLM-Guided Semantic-Based Ranking</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Scientific paper retrieval is essential for supporting literature discovery and research. While dense retrieval methods demonstrate effectiveness in general-purpose tasks, they often fail to capture fine-grained scientific concepts that are essential for accurate understanding of scientific queries. Recent studies also use large language models (LLMs) for query understanding; however, these methods often lack grounding in corpus-specific knowledge and may generate unreliable or unfaithful content. To overcome these limitations, we propose SemRank, an effective and efficient paper retrieval framework that combines LLM-guided query understanding with a concept-based semantic index. Each paper is indexed using multi-granular scientific concepts, including general research topics and detailed key phrases. At query time, an LLM identifies core concepts derived from the corpus to explicitly capture the query's information need. These identified concepts enable precise semantic matching, significantly enhancing retrieval accuracy. Experiments show that SemRank consistently improves the performance of various base retrievers, surpasses strong existing LLM-based baselines, and remains highly efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21807v1">TabReason: A Reinforcement Learning-Enhanced Reasoning LLM for Explainable Tabular Data Prediction</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Predictive modeling on tabular data is the cornerstone of many real-world applications. Although gradient boosting machines and some recent deep models achieve strong performance on tabular data, they often lack interpretability. On the other hand, large language models (LLMs) have demonstrated powerful capabilities to generate human-like reasoning and explanations, but remain under-performed for tabular data prediction. In this paper, we propose a new approach that leverages reasoning-based LLMs, trained using reinforcement learning, to perform more accurate and explainable predictions on tabular data. Our method introduces custom reward functions that guide the model not only toward high prediction accuracy but also toward human-understandable reasons for its predictions. Experimental results show that our model achieves promising performance on financial benchmark datasets, outperforming most existing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21800v1">From Directions to Cones: Exploring Multidimensional Representations of Propositional Facts in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong conversational abilities but often generate falsehoods. Prior work suggests that the truthfulness of simple propositions can be represented as a single linear direction in a model's internal activations, but this may not fully capture its underlying geometry. In this work, we extend the concept cone framework, recently introduced for modeling refusal, to the domain of truth. We identify multi-dimensional cones that causally mediate truth-related behavior across multiple LLM families. Our results are supported by three lines of evidence: (i) causal interventions reliably flip model responses to factual statements, (ii) learned cones generalize across model architectures, and (iii) cone-based interventions preserve unrelated model behavior. These findings reveal the richer, multidirectional structure governing simple true/false propositions in LLMs and highlight concept cones as a promising tool for probing abstract behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14737v2">Dissecting the Ullman Variations with a SCALPEL: Why do LLMs fail at Trivial Alterations to the False Belief Task?</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Recent empirical results have sparked a debate about whether or not Large Language Models (LLMs) are capable of Theory of Mind (ToM). While some have found LLMs to be successful on ToM evaluations such as the False Belief task, others have shown that their performance is not robust against trivial alterations to stimuli. In this paper, we introduce SCALPEL -- a technique to incrementally modify stimuli to test different specific hypotheses about why LLMs fail -- and apply this method to the "transparent-access" modification of the unexpected contents task. Our results suggest that LLMs often do poorly because they fail to make essential common-sense inferences, such as that seeing a transparent container implies recognizing its contents. We conclude that while modern LLMs go beyond mere pattern matching, they still fall short of robust human-like ToM. We argue that SCALPEL can help cognitive scientists examine LLMs' capabilities in finer detail and provide insight into alternative mechanisms by which tasks that are used to assess human cognition might be completed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21784v1">Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 Accepted to ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Safety reasoning is a recent paradigm where LLMs reason over safety policies before generating responses, thereby mitigating limitations in existing safety measures such as over-refusal and jailbreak vulnerabilities. However, implementing this paradigm is challenging due to the resource-intensive process of creating high-quality policy-embedded chain-of-thought (CoT) datasets while ensuring reasoning remains accurate and free from hallucinations or policy conflicts. To tackle this, we propose AIDSAFE: Agentic Iterative Deliberation for Safety Reasoning, a novel data generation recipe that leverages multi-agent deliberation to iteratively expand reasoning on safety policies. A data refiner stage in AIDSAFE ensures high-quality outputs by eliminating repetitive, redundant, and deceptive thoughts. AIDSAFE-generated CoTs provide a strong foundation for supervised fine-tuning (SFT)-based safety training. Additionally, to address the need of preference data in alignment stages, such as DPO training, we introduce a supplemental recipe that uses belief augmentation to create distinct selected and rejected CoT samples. Our evaluations demonstrate that AIDSAFE-generated CoTs achieve superior policy adherence and reasoning quality. Consequently, we show that fine-tuning open-source LLMs on these CoTs can significantly improve safety generalization and jailbreak robustness while maintaining acceptable utility and over-refusal accuracy. AIDSAFE-generated CoT datasets can be found here: https://huggingface.co/datasets/AmazonScience/AIDSAFE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08775v2">Layers at Similar Depths Generate Similar Activations Across LLM Architectures</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      How do the latent spaces used by independently-trained LLMs relate to one another? We study the nearest neighbor relationships induced by activations at different layers of 24 open-weight LLMs, and find that they 1) tend to vary from layer to layer within a model, and 2) are approximately shared between corresponding layers of different models. Claim 2 shows that these nearest neighbor relationships are not arbitrary, as they are shared across models, but Claim 1 shows that they are not "obvious" either, as there is no single set of nearest neighbor relationships that is universally shared. Together, these suggest that LLMs generate a progression of activation geometries from layer to layer, but that this entire progression is largely shared between models, stretched and squeezed to fit into different architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21775v1">DualSchool: How Reliable are LLMs for Optimization Education?</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Consider the following task taught in introductory optimization courses which addresses challenges articulated by the community at the intersection of (generative) AI and OR: generate the dual of a linear program. LLMs, being trained at web-scale, have the conversion process and many instances of Primal to Dual Conversion (P2DC) at their disposal. Students may thus reasonably expect that LLMs would perform well on the P2DC task. To assess this expectation, this paper introduces DualSchool, a comprehensive framework for generating and verifying P2DC instances. The verification procedure of DualSchool uses the Canonical Graph Edit Distance, going well beyond existing evaluation methods for optimization models, which exhibit many false positives and negatives when applied to P2DC. Experiments performed by DualSchool reveal interesting findings. Although LLMs can recite the conversion procedure accurately, state-of-the-art open LLMs fail to consistently produce correct duals. This finding holds even for the smallest two-variable instances and for derivative tasks, such as correctness, verification, and error classification. The paper also discusses the implications for educators, students, and the development of large reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16170v2">When Do LLMs Admit Their Mistakes? Understanding the Role of Model Belief in Retraction</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 Fixed typos
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) admit their mistakes when they should know better? In this work, we define the behavior of acknowledging errors in previously generated answers as "retraction" and aim to understand when and why LLMs choose to retract. We first construct model-specific datasets to evaluate whether a model will retract an incorrect answer that contradicts its own parametric knowledge. While LLMs are capable of retraction, they do so only infrequently. We demonstrate that retraction is closely tied to previously identified indicators of models' internal belief: models fail to retract wrong answers that they "believe" to be factually correct. Steering experiments further demonstrate that internal belief causally influences model retraction. In particular, when the model does not believe its answer, this not only encourages the model to attempt to verify the answer, but also alters attention behavior during self-verification. Finally, we demonstrate that simple supervised fine-tuning significantly improves retraction performance by helping the model learn more accurate internal beliefs. Code and datasets are available on https://github.com/ayyyq/llm-retraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21772v1">Calibrating LLM Confidence by Probing Perturbed Representation Stability</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Miscalibration in Large Language Models (LLMs) undermines their reliability, highlighting the need for accurate confidence estimation. We introduce CCPS (Calibrating LLM Confidence by Probing Perturbed Representation Stability), a novel method analyzing internal representational stability in LLMs. CCPS applies targeted adversarial perturbations to final hidden states, extracts features reflecting the model's response to these perturbations, and uses a lightweight classifier to predict answer correctness. CCPS was evaluated on LLMs from 8B to 32B parameters (covering Llama, Qwen, and Mistral architectures) using MMLU and MMLU-Pro benchmarks in both multiple-choice and open-ended formats. Our results show that CCPS significantly outperforms current approaches. Across four LLMs and three MMLU variants, CCPS reduces Expected Calibration Error by approximately 55% and Brier score by 21%, while increasing accuracy by 5 percentage points, Area Under the Precision-Recall Curve by 4 percentage points, and Area Under the Receiver Operating Characteristic Curve by 6 percentage points, all relative to the strongest prior method. CCPS delivers an efficient, broadly applicable, and more accurate solution for estimating LLM confidence, thereby improving their trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05522v4">User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 ACL'25 (Industry) Oral
    </div>
    <details class="paper-abstract">
      Exploration, the act of broadening user experiences beyond their established preferences, is challenging in large-scale recommendation systems due to feedback loops and limited signals on user exploration patterns. Large Language Models (LLMs) offer potential solutions by leveraging their world knowledge to recommend novel content outside these loops. A key challenge is aligning LLMs with user preferences while preserving their knowledge and reasoning. To enhance planning for new user interests using LLMs, this paper introduces a novel approach that combines hierarchical planning with LLM inference-time scaling. This method aims to improve recommendation relevancy without compromising novelty. We decouple novelty and user-alignment, training separate LLMs for each objective. We then scale up the novelty-focused LLM's inference and select the best-of-n predictions using the user-aligned LLM. Live experiments demonstrate efficacy, showing significant gains in both user satisfaction (measured by watch activity and active user counts) and exploration diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01013v2">Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Time series analysis provides essential insights for real-world system dynamics and informs downstream decision-making, yet most existing methods often overlook the rich contextual signals present in auxiliary modalities. To bridge this gap, we introduce TimeXL, a multi-modal prediction framework that integrates a prototype-based time series encoder with three collaborating Large Language Models (LLMs) to deliver more accurate predictions and interpretable explanations. First, a multi-modal prototype-based encoder processes both time series and textual inputs to generate preliminary forecasts alongside case-based rationales. These outputs then feed into a prediction LLM, which refines the forecasts by reasoning over the encoder's predictions and explanations. Next, a reflection LLM compares the predicted values against the ground truth, identifying textual inconsistencies or noise. Guided by this feedback, a refinement LLM iteratively enhances text quality and triggers encoder retraining. This closed-loop workflow -- prediction, critique (reflect), and refinement -- continuously boosts the framework's performance and interpretability. Empirical evaluations on four real-world datasets demonstrate that TimeXL achieves up to 8.9\% improvement in AUC and produces human-centric, multi-modal explanations, highlighting the power of LLM-driven reasoning for time series prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14652v4">General-Reasoner: Advancing LLM Reasoning Across All Domains</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21740v1">Counterfactual Simulatability of LLM Explanations for Generation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21701v1">Do We Know What LLMs Don't Know? A Study of Consistency in Knowledge Probing</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      The reliability of large language models (LLMs) is greatly compromised by their tendency to hallucinate, underscoring the need for precise identification of knowledge gaps within LLMs. Various methods for probing such gaps exist, ranging from calibration-based to prompting-based methods. To evaluate these probing methods, in this paper, we propose a new process based on using input variations and quantitative metrics. Through this, we expose two dimensions of inconsistency in knowledge gap probing. (1) Intra-method inconsistency: Minimal non-semantic perturbations in prompts lead to considerable variance in detected knowledge gaps within the same probing method; e.g., the simple variation of shuffling answer options can decrease agreement to around 40%. (2) Cross-method inconsistency: Probing methods contradict each other on whether a model knows the answer. Methods are highly inconsistent -- with decision consistency across methods being as low as 7% -- even though the model, dataset, and prompt are all the same. These findings challenge existing probing methods and highlight the urgent need for perturbation-robust probing frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21693v1">MAKIEval: A Multilingual Automatic WiKidata-based Framework for Cultural Awareness Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used globally across many languages, but their English-centric pretraining raises concerns about cross-lingual disparities for cultural awareness, often resulting in biased outputs. However, comprehensive multilingual evaluation remains challenging due to limited benchmarks and questionable translation quality. To better assess these disparities, we introduce MAKIEval, an automatic multilingual framework for evaluating cultural awareness in LLMs across languages, regions, and topics. MAKIEval evaluates open-ended text generation, capturing how models express culturally grounded knowledge in natural language. Leveraging Wikidata's multilingual structure as a cross-lingual anchor, it automatically identifies cultural entities in model outputs and links them to structured knowledge, enabling scalable, language-agnostic evaluation without manual annotation or translation. We then introduce four metrics that capture complementary dimensions of cultural awareness: granularity, diversity, cultural specificity, and consensus across languages. We assess 7 LLMs developed from different parts of the world, encompassing both open-source and proprietary systems, across 13 languages, 19 countries and regions, and 6 culturally salient topics (e.g., food, clothing). Notably, we find that models tend to exhibit stronger cultural awareness in English, suggesting that English prompts more effectively activate culturally grounded knowledge. We publicly release our code and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21689v1">LLMPR: A Novel LLM-Driven Transfer Learning based Petition Ranking Model</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 28 pages, 5 figures, journal paper, submitted to AI and Law
    </div>
    <details class="paper-abstract">
      The persistent accumulation of unresolved legal cases, especially within the Indian judiciary, significantly hampers the timely delivery of justice. Manual methods of prioritizing petitions are often prone to inefficiencies and subjective biases further exacerbating delays. To address this issue, we propose LLMPR (Large Language Model-based Petition Ranking), an automated framework that utilizes transfer learning and machine learning to assign priority rankings to legal petitions based on their contextual urgency. Leveraging the ILDC dataset comprising 7,593 annotated petitions, we process unstructured legal text and extract features through various embedding techniques, including DistilBERT, LegalBERT, and MiniLM. These textual embeddings are combined with quantitative indicators such as gap days, rank scores, and word counts to train multiple machine learning models, including Random Forest, Decision Tree, XGBoost, LightGBM, and CatBoost. Our experiments demonstrate that Random Forest and Decision Tree models yield superior performance, with accuracy exceeding 99% and a Spearman rank correlation of 0.99. Notably, models using only numerical features achieve nearly optimal ranking results (R2 = 0.988, \r{ho} = 0.998), while LLM-based embeddings offer only marginal gains. These findings suggest that automated petition ranking can effectively streamline judicial workflows, reduce case backlog, and improve fairness in legal prioritization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09674v3">Probabilistic Reasoning with LLMs for k-anonymity Estimation</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 9 pages, preprint
    </div>
    <details class="paper-abstract">
      Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21684v1">Incentivizing Permissionless Distributed Learning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      We describe an incentive system for distributed deep learning of foundational models where peers are rewarded for contributions. The incentive system, \textit{Gauntlet}, has been deployed on the bittensor blockchain and used to train a 1.2B LLM with completely permissionless contributions of pseudo-gradients: no control over the users that can register or their hardware. \textit{Gauntlet} can be applied to any synchronous distributed training scheme that relies on aggregating updates or pseudo-gradients. We rely on a two-stage mechanism for fast filtering of peer uptime, reliability, and synchronization, combined with the core component that estimates the loss before and after individual pseudo-gradient contributions. We utilized an OpenSkill rating system to track competitiveness of pseudo-gradient scores across time. Finally, we introduce a novel mechanism to ensure peers on the network perform unique computations. Our live 1.2B run, which has paid out real-valued tokens to participants based on the value of their contributions, yielded a competitive (on a per-iteration basis) 1.2B model that demonstrates the utility of our incentive system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21505v1">How does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Multilingual Alignment is an effective and representative paradigm to enhance LLMs' multilingual capabilities, which transfers the capabilities from the high-resource languages to the low-resource languages. Meanwhile, some researches on language-specific neurons reveal that there are language-specific neurons that are selectively activated in LLMs when processing different languages. This provides a new perspective to analyze and understand LLMs' mechanisms more specifically in multilingual scenarios. In this work, we propose a new finer-grained neuron identification algorithm, which detects language neurons~(including language-specific neurons and language-related neurons) and language-agnostic neurons. Furthermore, based on the distributional characteristics of different types of neurons, we divide the LLMs' internal process for multilingual inference into four parts: (1) multilingual understanding, (2) shared semantic space reasoning, (3) multilingual output space transformation, and (4) vocabulary space outputting. Additionally, we systematically analyze the models before and after alignment with a focus on different types of neurons. We also analyze the phenomenon of ''Spontaneous Multilingual Alignment''. Overall, our work conducts a comprehensive investigation based on different types of neurons, providing empirical results and valuable insights for better understanding multilingual alignment and multilingual capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21503v1">Silence is Not Consensus: Disrupting Agreement Bias in Multi-Agent LLMs via Catfish Agent for Clinical Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong potential in clinical question answering, with recent multi-agent frameworks further improving diagnostic accuracy via collaborative reasoning. However, we identify a recurring issue of Silent Agreement, where agents prematurely converge on diagnoses without sufficient critical analysis, particularly in complex or ambiguous cases. We present a new concept called Catfish Agent, a role-specialized LLM designed to inject structured dissent and counter silent agreement. Inspired by the ``catfish effect'' in organizational psychology, the Catfish Agent is designed to challenge emerging consensus to stimulate deeper reasoning. We formulate two mechanisms to encourage effective and context-aware interventions: (i) a complexity-aware intervention that modulates agent engagement based on case difficulty, and (ii) a tone-calibrated intervention articulated to balance critique and collaboration. Evaluations on nine medical Q&A and three medical VQA benchmarks show that our approach consistently outperforms both single- and multi-agent LLMs frameworks, including leading commercial models such as GPT-4o and DeepSeek-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21486v1">Robust Hypothesis Generation: LLM-Automated Language Bias for Inductive Logic Programming</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Automating robust hypothesis generation in open environments is pivotal for AI cognition. We introduce a novel framework integrating a multi-agent system, powered by Large Language Models (LLMs), with Inductive Logic Programming (ILP). Our system's LLM agents autonomously define a structured symbolic vocabulary (predicates) and relational templates , i.e., \emph{language bias} directly from raw textual data. This automated symbolic grounding (the construction of the language bias), traditionally an expert-driven bottleneck for ILP, then guides the transformation of text into facts for an ILP solver, which inductively learns interpretable rules. This approach overcomes traditional ILP's reliance on predefined symbolic structures and the noise-sensitivity of pure LLM methods. Extensive experiments in diverse, challenging scenarios validate superior performance, paving a new path for automated, explainable, and verifiable hypothesis generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21471v1">Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 30 pages, 9 figures. Code and data are available at https://github.com/THUNLP-MT/ExtAgents
    </div>
    <details class="paper-abstract">
      With the rapid advancement of post-training techniques for reasoning and information seeking, large language models (LLMs) can incorporate a large quantity of retrieved knowledge to solve complex tasks. However, the limited context window of LLMs obstructs scaling the amount of external knowledge input, prohibiting further improvement, especially for tasks requiring significant amount of external knowledge. Existing context window extension methods inevitably cause information loss. LLM-based multi-agent methods emerge as a new paradigm to handle massive input in a distributional manner, where we identify two core bottlenecks in existing knowledge synchronization and reasoning processes. In this work, we develop a multi-agent framework, $\textbf{ExtAgents}$, to overcome the bottlenecks and enable better scalability in inference-time knowledge integration without longer-context training. Benchmarked with our enhanced multi-hop question answering test, $\textbf{$\boldsymbol{\infty}$Bench+}$, and other public test sets including long survey generation, ExtAgents significantly enhances the performance over existing non-training methods with the same amount of external knowledge input, regardless of whether it falls $\textit{within or exceeds the context window}$. Moreover, the method maintains high efficiency due to high parallelism. Further study in the coordination of LLM agents on increasing external knowledge input could benefit real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21458v1">Do LLMs Need to Think in One Language? Correlation between Latent Language and Task Performance</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to process information using a proficient internal language consistently, referred to as latent language, which may differ from the input or output languages. However, how the discrepancy between the latent language and the input and output language affects downstream task performance remains largely unexplored. While many studies research the latent language of LLMs, few address its importance in influencing task performance. In our study, we hypothesize that thinking in latent language consistently enhances downstream task performance. To validate this, our work varies the input prompt languages across multiple downstream tasks and analyzes the correlation between consistency in latent language and task performance. We create datasets consisting of questions from diverse domains such as translation and geo-culture, which are influenced by the choice of latent language. Experimental results across multiple LLMs on translation and geo-culture tasks, which are sensitive to the choice of language, indicate that maintaining consistency in latent language is not always necessary for optimal downstream task performance. This is because these models adapt their internal representations near the final layers to match the target language, reducing the impact of consistency on overall performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09192v2">Thinking beyond the anthropomorphic paradigm benefits LLM research</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Anthropomorphism, or the attribution of human traits to technology, is an automatic and unconscious response that occurs even in those with advanced technical expertise. In this position paper, we analyze hundreds of thousands of research articles to present empirical evidence of the prevalence and growth of anthropomorphic terminology in research on large language models (LLMs). We argue for challenging the deeper assumptions reflected in this terminology -- which, though often useful, may inadvertently constrain LLM development -- and broadening beyond them to open new pathways for understanding and improving LLMs. Specifically, we identify and examine five anthropomorphic assumptions that shape research across the LLM development lifecycle. For each assumption (e.g., that LLMs must use natural language for reasoning, or that they should be evaluated on benchmarks originally meant for humans), we demonstrate empirical, non-anthropomorphic alternatives that remain under-explored yet offer promising directions for LLM research and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19184v2">When Two LLMs Debate, Both Think They'll Win</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Can LLMs accurately adjust their confidence when facing opposition? Building on previous studies measuring calibration on static fact-based question-answering tasks, we evaluate Large Language Models (LLMs) in a dynamic, adversarial debate setting, uniquely combining two realistic factors: (a) a multi-turn format requiring models to update beliefs as new information emerges, and (b) a zero-sum structure to control for task-related uncertainty, since mutual high-confidence claims imply systematic overconfidence. We organized 60 three-round policy debates among ten state-of-the-art LLMs, with models privately rating their confidence (0-100) in winning after each round. We observed five concerning patterns: (1) Systematic overconfidence: models began debates with average initial confidence of 72.9% vs. a rational 50% baseline. (2) Confidence escalation: rather than reducing confidence as debates progressed, debaters increased their win probabilities, averaging 83% by the final round. (3) Mutual overestimation: in 61.7% of debates, both sides simultaneously claimed >=75% probability of victory, a logical impossibility. (4) Persistent self-debate bias: models debating identical copies increased confidence from 64.1% to 75.2%; even when explicitly informed their chance of winning was exactly 50%, confidence still rose (from 50.0% to 57.1%). (5) Misaligned private reasoning: models' private scratchpad thoughts sometimes differed from their public confidence ratings, raising concerns about faithfulness of chain-of-thought reasoning. These results suggest LLMs lack the ability to accurately self-assess or update their beliefs in dynamic, multi-turn tasks; a major concern as LLM outputs are deployed without careful review in assistant roles or agentic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01082v6">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 Oral presentation at ICLR 2025. Camera-ready version available at https://iclr.cc/virtual/2025/poster/30358
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. Popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures which lead to incoherent or repetitive outputs. We propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by using the top token's probability as a scaling factor. Our experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing show that min-p sampling improves both the quality and diversity of generated text across different model families (Mistral and Llama 3) and model sizes (1B to 123B parameters), especially at higher temperatures. Human evaluations further show a clear preference for min-p sampling, in both text quality and creativity. Min-p sampling has been adopted by popular open-source LLM frameworks, including Hugging Face Transformers, VLLM, and many others, highlighting its considerable impact on improving text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13010v2">Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced medical question-answering by leveraging extensive clinical data and medical literature. However, the rapid evolution of medical knowledge and the labor-intensive process of manually updating domain-specific resources pose challenges to the reliability of these systems. To address this, we introduce Agentic Medical Graph-RAG (AMG-RAG), a comprehensive framework that automates the construction and continuous updating of medical knowledge graphs, integrates reasoning, and retrieves current external evidence, such as PubMed and WikiSearch. By dynamically linking new findings and complex medical concepts, AMG-RAG not only improves accuracy but also enhances interpretability in medical queries. Evaluations on the MEDQA and MEDMCQA benchmarks demonstrate the effectiveness of AMG-RAG, achieving an F1 score of 74.1 percent on MEDQA and an accuracy of 66.34 percent on MEDMCQA, outperforming both comparable models and those 10 to 100 times larger. Notably, these improvements are achieved without increasing computational overhead, highlighting the critical role of automated knowledge graph generation and external evidence retrieval in delivering up-to-date, trustworthy medical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05159v2">A Lightweight Method to Disrupt Memorized Sequences in LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 26 pages, 3 figures
    </div>
    <details class="paper-abstract">
      As language models scale, their performance improves dramatically across a wide range of tasks, but so does their tendency to memorize and regurgitate parts of their training data verbatim. This tradeoff poses serious legal, ethical, and safety concerns, especially in real-world deployments. Existing mitigation techniques, such as differential privacy or model unlearning, often require retraining or access to internal weights making them impractical for most users. In this work, we introduce TokenSwap, a lightweight, post-hoc defense designed for realistic settings where the user can only access token-level outputs. Our key insight is that while large models are necessary for high task performance, small models (e.g., DistilGPT-2) are often sufficient to assign fluent, grammatically plausible probabilities to common function words - and crucially, they memorize far less. By selectively swapping token probabilities between models, TokenSwap preserves the capabilities of large models while reducing their propensity for verbatim reproduction. Evaluations on Pythia-6.9B and Llama-3-8B show up to a 10$\times$ drop in exact memorization with negligible task degradation. Our method offers a practical, accessible solution for mitigating memorized generation in deployed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11618v2">Benchmarking Spatiotemporal Reasoning in LLMs and Reasoning Models: Capabilities and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Spatiotemporal reasoning plays a key role in Cyber-Physical Systems (CPS). Despite advances in Large Language Models (LLMs) and Large Reasoning Models (LRMs), their capacity to reason about complex spatiotemporal signals remains underexplored. This paper proposes a hierarchical SpatioTemporal reAsoning benchmaRK, STARK, to systematically evaluate LLMs across three levels of reasoning complexity: state estimation (e.g., predicting field variables, localizing and tracking events in space and time), spatiotemporal reasoning over states (e.g., inferring spatial-temporal relationships), and world-knowledge-aware reasoning that integrates contextual and domain knowledge (e.g., intent prediction, landmark-aware navigation). We curate 26 distinct spatiotemporal tasks with diverse sensor modalities, comprising 14,552 challenges where models answer directly or by Python Code Interpreter. Evaluating 3 LRMs and 8 LLMs, we find LLMs achieve limited success in tasks requiring geometric reasoning (e.g., multilateration or triangulation), particularly as complexity increases. Surprisingly, LRMs show robust performance across tasks with various levels of difficulty, often competing or surpassing traditional first-principle-based methods. Our results show that in reasoning tasks requiring world knowledge, the performance gap between LLMs and LRMs narrows, with some LLMs even surpassing LRMs. However, the LRM o3 model continues to achieve leading performance across all evaluated tasks, a result attributed primarily to the larger size of the reasoning models. STARK motivates future innovations in model architectures and reasoning paradigms for intelligent CPS by providing a structured framework to identify limitations in the spatiotemporal reasoning of LLMs and LRMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21419v1">Diagnosing and Resolving Cloud Platform Instability with Multi-modal RAG LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 Published in EuroMLSys2025
    </div>
    <details class="paper-abstract">
      Today's cloud-hosted applications and services are complex systems, and a performance or functional instability can have dozens or hundreds of potential root causes. Our hypothesis is that by combining the pattern matching capabilities of modern AI tools with a natural multi-modal RAG LLM interface, problem identification and resolution can be simplified. ARCA is a new multi-modal RAG LLM system that targets this domain. Step-wise evaluations show that ARCA outperforms state-of-the-art alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21418v1">Autonomous Multi-Modal LLM Agents for Treatment Planning in Focused Ultrasound Ablation Surgery</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Focused Ultrasound Ablation Surgery (FUAS) has emerged as a promising non-invasive therapeutic modality, valued for its safety and precision. Nevertheless, its clinical implementation entails intricate tasks such as multimodal image interpretation, personalized dose planning, and real-time intraoperative decision-making processes that demand intelligent assistance to improve efficiency and reliability. We introduce FUAS-Agents, an autonomous agent system that leverages the multimodal understanding and tool-using capabilities of large language models (LLMs). By integrating patient profiles and MRI data, FUAS-Agents orchestrates a suite of specialized medical AI tools, including segmentation, treatment dose prediction, and clinical guideline retrieval, to generate personalized treatment plans comprising MRI image, dose parameters, and therapeutic strategies. We evaluate the system in a uterine fibroid treatment scenario. Human assessment by four senior FUAS experts indicates that 82.5%, 82.5%, 87.5%, and 97.5% of the generated plans were rated 4 or above (on a 5-point scale) in terms of completeness, accuracy, fluency, and clinical compliance, respectively. These results demonstrate the potential of LLM-driven agents in enhancing decision-making across complex clinical workflows, and exemplify a translational paradigm that combines general-purpose models with specialized expert systems to solve practical challenges in vertical healthcare domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20993v2">Efficiently Scaling LLM Reasoning with Certaindex</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Test-time reasoning algorithms such as chain-of-thought, self-consistency, and MCTS enhance LLM problem-solving but can wastefully generate many tokens without improving accuracy. At the same time, we observe that these algorithms exhibit answer stabilization: their intermediate solutions often cease to change after a certain point, and further investment of compute does not change their final answer. To quantify this phenomenon, we introduce Certaindex, an algorithm-agnostic metric measuring this evolving stability, signaling when further computation is unlikely to alter the final result. Certaindex is lightweight, can accelerate reasoning program inference via early exit, and further enables dynamic token allocation, gang scheduling, and many opportunities when integrated with real-world LLM serving systems. To quantify real-world benefits, we built Certaindex as a scheduler into Dynasor, our reasoning-aware LLM serving system, and demonstrate up to 50% compute savings and 3.3x higher throughput in real workloads with no accuracy drop. Our code is available at https://github.com/hao-ai-lab/Dynasor.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09598v2">How to Protect Yourself from 5G Radiation? Investigating LLM Responses to Implicit Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are widely deployed in diverse scenarios, the extent to which they could tacitly spread misinformation emerges as a critical safety concern. Current research primarily evaluates LLMs on explicit false statements, overlooking how misinformation often manifests subtly as unchallenged premises in real-world interactions. We curated EchoMist, the first comprehensive benchmark for implicit misinformation, where false assumptions are embedded in the query to LLMs. EchoMist targets circulated, harmful, and ever-evolving implicit misinformation from diverse sources, including realistic human-AI conversations and social media interactions. Through extensive empirical studies on 15 state-of-the-art LLMs, we find that current models perform alarmingly poorly on this task, often failing to detect false premises and generating counterfactual explanations. We also investigate two mitigation methods, i.e., Self-Alert and RAG, to enhance LLMs' capability to counter implicit misinformation. Our findings indicate that EchoMist remains a persistent challenge and underscore the critical need to safeguard against the risk of implicit misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2502.02810v2">Mol-LLM: Multimodal Generalist Molecular LLM with Improved Graph Utilization</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have led to models that tackle diverse molecular tasks, such as chemical reaction prediction and molecular property prediction. Large-scale molecular instruction-tuning datasets have enabled sequence-only (e.g., SMILES or SELFIES) generalist molecular LLMs, and researchers are now exploring multimodal approaches that incorporate molecular structural information for further gains. However, a genuinely multimodal, generalist LLM that covers a broad spectrum of molecular tasks has yet to be fully investigated. We observe that naive next token prediction training ignores graph-structural information, limiting an LLM's ability to exploit molecular graphs. To address this, we propose (i) Molecular structure Preference Optimization (MolPO), which facilitates graph usage by optimizing preferences between pairs of correct and perturbed molecular structures, and (ii) an advanced graph encoder with a tailored pre-training strategy to improve the effect of graph utilization by MolPO. Building on these contributions, we introduce Mol-LLM, the first multimodal generalist model that (a) handles a broad spectrum of molecular tasks among molecular LLMs, (b) explicitly leverages molecular-structure information, and (c) takes advantage of extensive instruction tuning. Mol-LLM attains state-of-the-art or comparable results across the most comprehensive molecular-LLM benchmark-even on out-of-distribution datasets for reaction and property prediction, where it surpasses prior generalist molecular LLMs by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.14434v2">LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Automated feature engineering plays a critical role in improving predictive model performance for tabular learning tasks. Traditional automated feature engineering methods are limited by their reliance on pre-defined transformations within fixed, manually designed search spaces, often neglecting domain knowledge. Recent advances using Large Language Models (LLMs) have enabled the integration of domain knowledge into the feature engineering process. However, existing LLM-based approaches use direct prompting or rely solely on validation scores for feature selection, failing to leverage insights from prior feature discovery experiments or establish meaningful reasoning between feature generation and data-driven performance. To address these challenges, we propose LLM-FE, a novel framework that combines evolutionary search with the domain knowledge and reasoning capabilities of LLMs to automatically discover effective features for tabular learning tasks. LLM-FE formulates feature engineering as a program search problem, where LLMs propose new feature transformation programs iteratively, and data-driven feedback guides the search process. Our results demonstrate that LLM-FE consistently outperforms state-of-the-art baselines, significantly enhancing the performance of tabular prediction models across diverse classification and regression benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19504v1">DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Code is available at https://github.com/UNITES-Lab/DOGe
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) represent substantial intellectual and economic investments, yet their effectiveness can inadvertently facilitate model imitation via knowledge distillation (KD).In practical scenarios, competitors can distill proprietary LLM capabilities by simply observing publicly accessible outputs, akin to reverse-engineering a complex performance by observation alone. Existing protective methods like watermarking only identify imitation post-hoc, while other defenses assume the student model mimics the teacher's internal logits, rendering them ineffective against distillation purely from observed output text. This paper confronts the challenge of actively protecting LLMs within the realistic constraints of API-based access. We introduce an effective and efficient Defensive Output Generation (DOGe) strategy that subtly modifies the output behavior of an LLM. Its outputs remain accurate and useful for legitimate users, yet are designed to be misleading for distillation, significantly undermining imitation attempts. We achieve this by fine-tuning only the final linear layer of the teacher LLM with an adversarial loss. This targeted training approach anticipates and disrupts distillation attempts during inference time. Our experiments show that, while preserving or even improving the original performance of the teacher model, student models distilled from the defensively generated teacher outputs demonstrate catastrophically reduced performance, demonstrating our method's effectiveness as a practical safeguard against KD-based model imitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17070v2">Robo-Troj: Attacking LLM-based Task Planners</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19489v1">Benchmarking and Enhancing LLM Agents in Localizing Linux Kernel Bugs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      The Linux kernel is a critical system, serving as the foundation for numerous systems. Bugs in the Linux kernel can cause serious consequences, affecting billions of users. Fault localization (FL), which aims at identifying the buggy code elements in software, plays an essential role in software quality assurance. While recent LLM agents have achieved promising accuracy in FL on recent benchmarks like SWE-bench, it remains unclear how well these methods perform in the Linux kernel, where FL is much more challenging due to the large-scale code base, limited observability, and diverse impact factors. In this paper, we introduce LinuxFLBench, a FL benchmark constructed from real-world Linux kernel bugs. We conduct an empirical study to assess the performance of state-of-the-art LLM agents on the Linux kernel. Our initial results reveal that existing agents struggle with this task, achieving a best top-1 accuracy of only 41.6% at file level. To address this challenge, we propose LinuxFL$^+$, an enhancement framework designed to improve FL effectiveness of LLM agents for the Linux kernel. LinuxFL$^+$ substantially improves the FL accuracy of all studied agents (e.g., 7.2% - 11.2% accuracy increase) with minimal costs. Data and code are available at https://github.com/FudanSELab/LinuxFLBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19484v1">CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they often exhibit a specific cultural biases, neglecting the values and linguistic diversity of low-resource regions. This cultural bias not only undermines universal equality, but also risks reinforcing stereotypes and perpetuating discrimination. To address this, we propose CulFiT, a novel culturally-aware training paradigm that leverages multilingual data and fine-grained reward modeling to enhance cultural sensitivity and inclusivity. Our approach synthesizes diverse cultural-related questions, constructs critique data in culturally relevant languages, and employs fine-grained rewards to decompose cultural texts into verifiable knowledge units for interpretable evaluation. We also introduce GlobalCultureQA, a multilingual open-ended question-answering dataset designed to evaluate culturally-aware responses in a global context. Extensive experiments on three existing benchmarks and our GlobalCultureQA demonstrate that CulFiT achieves state-of-the-art open-source model performance in cultural alignment and general reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19481v1">Win Fast or Lose Slow: Balancing Speed and Accuracy in Latency-Sensitive Decisions of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance across diverse reasoning and generation tasks, and are increasingly deployed as agents in dynamic environments such as code generation and recommendation systems. However, many real-world applications, such as high-frequency trading and real-time competitive gaming, require decisions under strict latency constraints, where faster responses directly translate into higher rewards. Despite the importance of this latency quality trade off, it remains underexplored in the context of LLM based agents. In this work, we present the first systematic study of this trade off in real time decision making tasks. To support our investigation, we introduce two new benchmarks: HFTBench, a high frequency trading simulation, and StreetFighter, a competitive gaming platform. Our analysis reveals that optimal latency quality balance varies by task, and that sacrificing quality for lower latency can significantly enhance downstream performance. To address this, we propose FPX, an adaptive framework that dynamically selects model size and quantization level based on real time demands. Our method achieves the best performance on both benchmarks, improving win rate by up to 80% in Street Fighter and boosting daily yield by up to 26.52% in trading, underscoring the need for latency aware evaluation and deployment strategies for LLM based agents. These results demonstrate the critical importance of latency aware evaluation and deployment strategies for real world LLM based agents. Our benchmarks are available at Latency Sensitive Benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17135v2">When can isotropy help adapt LLMs' next word prediction to numerical domains?</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Recent studies have shown that vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black-box and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numeric downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, we consider a log-linear model for LLMs in which numeric data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). We demonstrate that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, we show how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numeric data and model architecture could have different impacts on isotropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19473v1">Improving Recommendation Fairness without Sensitive Attributes Using Multi-Persona LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 18 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Despite the success of recommender systems in alleviating information overload, fairness issues have raised concerns in recent years, potentially leading to unequal treatment for certain user groups. While efforts have been made to improve recommendation fairness, they often assume that users' sensitive attributes are available during model training. However, collecting sensitive information can be difficult, especially on platforms that involve no personal information disclosure. Therefore, we aim to improve recommendation fairness without any access to sensitive attributes. However, this is a non-trivial task because uncovering latent sensitive patterns from complicated user behaviors without explicit sensitive attributes can be difficult. Consequently, suboptimal estimates of sensitive distributions can hinder the fairness training process. To address these challenges, leveraging the remarkable reasoning abilities of Large Language Models (LLMs), we propose a novel LLM-enhanced framework for Fair recommendation withOut Sensitive Attributes (LLMFOSA). A Multi-Persona Sensitive Information Inference module employs LLMs with distinct personas that mimic diverse human perceptions to infer and distill sensitive information. Furthermore, a Confusion-Aware Sensitive Representation Learning module incorporates inference results and rationales to develop robust sensitive representations, considering the mislabeling confusion and collective consensus among agents. The model is then optimized by a formulated mutual information objective. Extensive experiments on two public datasets validate the effectiveness of LLMFOSA in improving fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11393v2">HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in commonsense reasoning; however, some variations in questions can trigger incorrect responses. Do these models truly understand commonsense knowledge, or just memorize expression patterns? To investigate this question, we present the first extensive robustness evaluation of LLMs in commonsense reasoning. We introduce HellaSwag-Pro, a large-scale bilingual benchmark consisting of 11,200 cases, by designing and compiling seven types of question variants. To construct this benchmark, we propose a two-stage method to develop Chinese HellaSwag, a finely annotated dataset comprising 12,000 instances across 56 categories. We conduct extensive experiments on 41 representative LLMs, revealing that these LLMs are far from robust in commonsense reasoning. Furthermore, this robustness varies depending on the language in which the LLM is tested. This work establishes a high-quality evaluation benchmark, with extensive experiments offering valuable insights to the community in commonsense reasoning for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19466v1">Origin Tracer: A Method for Detecting LoRA Fine-Tuning Origins in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance, their deployment often involves fine-tuning to enhance performance on specific downstream tasks. However, this customization is sometimes accompanied by misleading claims about the origins, raising significant concerns about transparency and trust within the open-source community. Existing model verification techniques typically assess functional, representational, and weight similarities. However, these approaches often struggle against obfuscation techniques, such as permutations and scaling transformations. To address this limitation, we propose a novel detection method Origin-Tracer that rigorously determines whether a model has been fine-tuned from a specified base model. This method includes the ability to extract the LoRA rank utilized during the fine-tuning process, providing a more robust verification framework. This framework is the first to provide a formalized approach specifically aimed at pinpointing the sources of model fine-tuning. We empirically validated our method on thirty-one diverse open-source models under conditions that simulate real-world obfuscation scenarios. We empirically analyze the effectiveness of our framework and finally, discuss its limitations. The results demonstrate the effectiveness of our approach and indicate its potential to establish new benchmarks for model verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19464v1">LLMs as Better Recommenders with Natural Language Collaborative Signals: A Self-Assessing Retrieval Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Incorporating collaborative information (CI) effectively is crucial for leveraging LLMs in recommendation tasks. Existing approaches often encode CI using soft tokens or abstract identifiers, which introduces a semantic misalignment with the LLM's natural language pretraining and hampers knowledge integration. To address this, we propose expressing CI directly in natural language to better align with LLMs' semantic space. We achieve this by retrieving a curated set of the most relevant user behaviors in natural language form. However, identifying informative CI is challenging due to the complexity of similarity and utility assessment. To tackle this, we introduce a Self-assessing COllaborative REtrieval framework (SCORE) following the retrieve-rerank paradigm. First, a Collaborative Retriever (CAR) is developed to consider both collaborative patterns and semantic similarity. Then, a Self-assessing Reranker (SARE) leverages LLMs' own reasoning to assess and prioritize retrieved behaviors. Finally, the selected behaviors are prepended to the LLM prompt as natural-language CI to guide recommendation. Extensive experiments on two public datasets validate the effectiveness of SCORE in improving LLM-based recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19457v1">BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Project Page: https://hithink-research.github.io/BizFinBench/
    </div>
    <details class="paper-abstract">
      Large language models excel in general tasks, yet assessing their reliability in logic-heavy, precision-critical domains like finance, law, and healthcare remains challenging. To address this, we introduce BizFinBench, the first benchmark specifically designed to evaluate LLMs in real-world financial applications. BizFinBench consists of 6,781 well-annotated queries in Chinese, spanning five dimensions: numerical calculation, reasoning, information extraction, prediction recognition, and knowledge-based question answering, grouped into nine fine-grained categories. The benchmark includes both objective and subjective metrics. We also introduce IteraJudge, a novel LLM evaluation method that reduces bias when LLMs serve as evaluators in objective metrics. We benchmark 25 models, including both proprietary and open-source systems. Extensive experiments show that no model dominates across all tasks. Our evaluation reveals distinct capability patterns: (1) In Numerical Calculation, Claude-3.5-Sonnet (63.18) and DeepSeek-R1 (64.04) lead, while smaller models like Qwen2.5-VL-3B (15.92) lag significantly; (2) In Reasoning, proprietary models dominate (ChatGPT-o3: 83.58, Gemini-2.0-Flash: 81.15), with open-source models trailing by up to 19.49 points; (3) In Information Extraction, the performance spread is the largest, with DeepSeek-R1 scoring 71.46, while Qwen3-1.7B scores 11.23; (4) In Prediction Recognition, performance variance is minimal, with top models scoring between 39.16 and 50.00. We find that while current LLMs handle routine finance queries competently, they struggle with complex scenarios requiring cross-concept reasoning. BizFinBench offers a rigorous, business-aligned benchmark for future research. The code and dataset are available at https://github.com/HiThink-Research/BizFinBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19436v1">Task Memory Engine: Spatial Memory for Robust Multi-Step LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Under review. 9 pages main content, 15 pages appendix, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) falter in multi-step interactions -- often hallucinating, repeating actions, or misinterpreting user corrections -- due to reliance on linear, unstructured context. This fragility stems from the lack of persistent memory to track evolving goals and task dependencies, undermining trust in autonomous agents. We introduce the Task Memory Engine (TME), a modular memory controller that transforms existing LLMs into robust, revision-aware agents without fine-tuning. TME implements a spatial memory framework that replaces flat context with graph-based structures to support consistent, multi-turn reasoning. Departing from linear concatenation and ReAct-style prompting, TME builds a dynamic task graph -- either a tree or directed acyclic graph (DAG) -- to map user inputs to subtasks, align them with prior context, and enable dependency-tracked revisions. Its Task Representation and Intent Management (TRIM) component models task semantics and user intent to ensure accurate interpretation. Across four multi-turn scenarios-trip planning, cooking, meeting scheduling, and shopping cart editing -- TME eliminates 100% of hallucinations and misinterpretations in three tasks, and reduces hallucinations by 66.7% and misinterpretations by 83.3% across 27 user turns, outperforming ReAct. TME's modular design supports plug-and-play deployment and domain-specific customization, adaptable to both personal assistants and enterprise automation. We release TME's codebase, benchmarks, and components as open-source resources, enabling researchers to develop reliable LLM agents. TME's scalable architecture addresses a critical gap in agent performance across complex, interactive settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04856v2">One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19433v1">Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Accepted by ICML2025 as Poster
    </div>
    <details class="paper-abstract">
      Post-training compression reduces the computational and memory costs of large language models (LLMs), enabling resource-efficient deployment. However, existing compression benchmarks only focus on language modeling (e.g., perplexity) and natural language understanding tasks (e.g., GLUE accuracy), ignoring the agentic capabilities - workflow, tool use/function call, long-context understanding and real-world application. We introduce the Agent Compression Benchmark (ACBench), the first comprehensive benchmark for evaluating how compression impacts LLMs' agentic abilities. ACBench spans (1) 12 tasks across 4 capabilities (e.g., WorfBench for workflow generation, Needle-in-Haystack for long-context retrieval), (2) quantization (GPTQ, AWQ) and pruning (Wanda, SparseGPT), and (3) 15 models, including small (Gemma-2B), standard (Qwen2.5 7B-32B), and distilled reasoning LLMs (DeepSeek-R1-Distill). Our experiments reveal compression tradeoffs: 4-bit quantization preserves workflow generation and tool use (1%-3% drop) but degrades real-world application accuracy by 10%-15%. We introduce ERank, Top-k Ranking Correlation and Energy to systematize analysis. ACBench provides actionable insights for optimizing LLM compression in agentic scenarios. The code can be found in https://github.com/pprp/ACBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11478v3">Each Graph is a New Language: Graph Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Recent efforts leverage Large Language Models (LLMs) for modeling text-attributed graph structures in node classification tasks. These approaches describe graph structures for LLMs to understand or aggregate LLM-generated textual attribute embeddings through graph structure. However, these approaches face two main limitations in modeling graph structures with LLMs. (i) Graph descriptions become verbose in describing high-order graph structure. (ii) Textual attributes alone do not contain adequate graph structure information. It is challenging to model graph structure concisely and adequately with LLMs. LLMs lack built-in mechanisms to model graph structures directly. They also struggle with complex long-range dependencies between high-order nodes and target nodes. Inspired by the observation that LLMs pre-trained on one language can achieve exceptional performance on another with minimal additional training, we propose \textbf{G}raph-\textbf{D}efined \textbf{L}anguage for \textbf{L}arge \textbf{L}anguage \textbf{M}odel (GDL4LLM). This novel framework enables LLMs to transfer their powerful language understanding capabilities to graph-structured data. GDL4LLM translates graphs into a graph language corpus instead of graph descriptions and pre-trains LLMs on this corpus to adequately understand graph structures. During fine-tuning, this corpus describes the structural information of target nodes concisely with only a few tokens. By treating graphs as a new language, GDL4LLM enables LLMs to model graph structures adequately and concisely for node classification tasks. Extensive experiments on three real-world datasets demonstrate that GDL4LLM outperforms description-based and textual attribute embeddings-based baselines by efficiently modeling different orders of graph structure with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19419v1">It's Not Just Labeling" -- A Research on LLM Generated Feedback Interpretability and Image Labeling Sketch Features</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      The quality of training data is critical to the performance of machine learning applications in domains like transportation, healthcare, and robotics. Accurate image labeling, however, often relies on time-consuming, expert-driven methods with limited feedback. This research introduces a sketch-based annotation approach supported by large language models (LLMs) to reduce technical barriers and enhance accessibility. Using a synthetic dataset, we examine how sketch recognition features relate to LLM feedback metrics, aiming to improve the reliability and interpretability of LLM-assisted labeling. We also explore how prompting strategies and sketch variations influence feedback quality. Our main contribution is a sketch-based virtual assistant that simplifies annotation for non-experts and advances LLM-driven labeling tools in terms of scalability, accessibility, and explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19410v1">Self-Reflective Planning with Knowledge Graphs: Enhancing LLM Reasoning Reliability for Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated remarkable capabilities in natural language processing tasks, yet they remain prone to hallucinations when reasoning with insufficient internal knowledge. While integrating LLMs with knowledge graphs (KGs) provides access to structured, verifiable information, existing approaches often generate incomplete or factually inconsistent reasoning paths. To this end, we propose Self-Reflective Planning (SRP), a framework that synergizes LLMs with KGs through iterative, reference-guided reasoning. Specifically, given a question and topic entities, SRP first searches for references to guide planning and reflection. In the planning process, it checks initial relations and generates a reasoning path. After retrieving knowledge from KGs through a reasoning path, it implements iterative reflection by judging the retrieval result and editing the reasoning path until the answer is correctly retrieved. Extensive experiments on three public datasets demonstrate that SRP surpasses various strong baselines and further underscore its reliable reasoning ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19405v1">CoTGuard: Using Chain-of-Thought Triggering for Copyright Protection in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 18 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve into autonomous agents capable of collaborative reasoning and task execution, multi-agent LLM systems have emerged as a powerful paradigm for solving complex problems. However, these systems pose new challenges for copyright protection, particularly when sensitive or copyrighted content is inadvertently recalled through inter-agent communication and reasoning. Existing protection techniques primarily focus on detecting content in final outputs, overlooking the richer, more revealing reasoning processes within the agents themselves. In this paper, we introduce CoTGuard, a novel framework for copyright protection that leverages trigger-based detection within Chain-of-Thought (CoT) reasoning. Specifically, we can activate specific CoT segments and monitor intermediate reasoning steps for unauthorized content reproduction by embedding specific trigger queries into agent prompts. This approach enables fine-grained, interpretable detection of copyright violations in collaborative agent scenarios. We evaluate CoTGuard on various benchmarks in extensive experiments and show that it effectively uncovers content leakage with minimal interference to task performance. Our findings suggest that reasoning-level monitoring offers a promising direction for safeguarding intellectual property in LLM-based agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18791v3">Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      The surge of LLM studies makes synthesizing their findings challenging. Analysis of experimental results from literature can uncover important trends across studies, but the time-consuming nature of manual data extraction limits its use. Our study presents a semi-automated approach for literature analysis that accelerates data extraction using LLMs. It automatically identifies relevant arXiv papers, extracts experimental results and related attributes, and organizes them into a structured dataset, LLMEvalDB. We then conduct an automated literature analysis of frontier LLMs, reducing the effort of paper surveying and data extraction by more than 93% compared to manual approaches. We validate LLMEvalDB by showing that it reproduces key findings from a recent manual analysis of Chain-of-Thought (CoT) reasoning and also uncovers new insights that go beyond it, showing, for example, that in-context examples benefit coding & multimodal tasks but offer limited gains in math reasoning tasks compared to zero-shot CoT. Our automatically updatable dataset enables continuous tracking of target models by extracting evaluation studies as new data becomes available. Through LLMEvalDB and empirical analysis, we provide insights into LLMs while facilitating ongoing literature analyses of their behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12632v3">What External Knowledge is Preferred by LLMs? Characterizing and Exploring Chain of Evidence in Imperfect Context for Multi-Hop QA</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Incorporating external knowledge has emerged as a promising way to mitigate outdated knowledge and hallucinations in LLM. However, external knowledge is often imperfect, encompassing substantial extraneous or even inaccurate content, which interferes with the LLM's utilization of useful knowledge in the context. This paper seeks to characterize the features of preferred external knowledge and perform empirical studies in imperfect contexts. Inspired by the chain of evidence (CoE), we characterize that the knowledge preferred by LLMs should maintain both relevance to the question and mutual support among the textual pieces. Accordingly, we propose a CoE discrimination approach and conduct a comparative analysis between CoE and Non-CoE samples across significance, deceptiveness, and robustness, revealing the LLM's preference for external knowledge that aligns with CoE features. Furthermore, we selected three representative tasks (RAG-based multi-hop QA, external knowledge poisoning and poisoning defense), along with corresponding SOTA or prevalent baselines. By integrating CoE features, the variants achieved significant improvements over the original baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16254v2">Reassessing Collaborative Writing Theories and Frameworks in the Age of LLMs: What Still Applies and What We Must Leave Behind</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      In this paper, we conduct a critical review of existing theories and frameworks on human-human collaborative writing to assess their relevance to the current human-AI paradigm in professional contexts, and draw seven insights along with design implications for human-AI collaborative writing tools. We found that, as LLMs nudge the writing process more towards an empirical "trial and error" process analogous to prototyping, the non-linear cognitive process of writing will stay the same, but more rigor will be required for revision methodologies. This shift would shed further light on the importance of coherence support, but the large language model (LLM)'s unprecedented semantic capabilities can bring novel approaches to this ongoing challenge. We argue that teamwork-related factors such as group awareness, consensus building and authorship - which have been central in human-human collaborative writing studies - should not apply to the human-AI paradigm due to excessive anthropomorphism. With the LLM's text generation capabilities becoming essentially indistinguishable from human-written ones, we are entering an era where, for the first time in the history of computing, we are engaging in collaborative writing with AI at workplaces on a daily basis. We aim to bring theoretical grounding and practical design guidance to the interaction designs of human-AI collaborative writing, with the goal of enhancing future human-AI writing software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18152v2">Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Github:https://github.com/mbzuai-oryx/FannOrFlop, Dataset:https://huggingface.co/datasets/omkarthawakar/FannOrFlop
    </div>
    <details class="paper-abstract">
      Arabic poetry is one of the richest and most culturally rooted forms of expression in the Arabic language, known for its layered meanings, stylistic diversity, and deep historical continuity. Although large language models (LLMs) have demonstrated strong performance across languages and tasks, their ability to understand Arabic poetry remains largely unexplored. In this work, we introduce \emph{Fann or Flop}, the first benchmark designed to assess the comprehension of Arabic poetry by LLMs in 12 historical eras, covering 14 core poetic genres and a variety of metrical forms, from classical structures to contemporary free verse. The benchmark comprises a curated corpus of poems with explanations that assess semantic understanding, metaphor interpretation, prosodic awareness, and cultural context. We argue that poetic comprehension offers a strong indicator for testing how good the LLM understands classical Arabic through Arabic poetry. Unlike surface-level tasks, this domain demands deeper interpretive reasoning and cultural sensitivity. Our evaluation of state-of-the-art LLMs shows that most models struggle with poetic understanding despite strong results on standard Arabic benchmarks. We release "Fann or Flop" along with the evaluation suite as an open-source resource to enable rigorous evaluation and advancement for Arabic language models. Code is available at: https://github.com/mbzuai-oryx/FannOrFlop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20573v1">Collision- and Reachability-Aware Multi-Robot Control with Grounded LLM Planners</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance in various robot control tasks. However, their deployment in real-world applications remains constrained. Even state-ofthe-art LLMs, such as GPT-o4mini, frequently produce invalid action plans that violate physical constraints, such as directing a robot to an unreachable location or causing collisions between robots. This issue primarily arises from a lack of awareness of these physical constraints during the reasoning process. To address this issue, we propose a novel framework that integrates reinforcement learning with verifiable rewards (RLVR) to incentivize knowledge of physical constraints into LLMs to induce constraints-aware reasoning during plan generation. In this approach, only valid action plans that successfully complete a control task receive positive rewards. We applied our method to two small-scale LLMs: a non-reasoning Qwen2.5-3B-Instruct and a reasoning Qwen3-4B. The experiment results demonstrate that constraint-aware small LLMs largely outperform large-scale models without constraints, grounded on both the BoxNet task and a newly developed BoxNet3D environment built using MuJoCo. This work highlights the effectiveness of grounding even small LLMs with physical constraints to enable scalable and efficient multi-robot control in complex, physically constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20561v1">Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained via Reinforcement Learning (RL) have exhibited strong reasoning capabilities and emergent reflective behaviors, such as backtracking and error correction. However, conventional Markovian RL confines exploration to the training phase to learn an optimal deterministic policy and depends on the history contexts only through the current state. Therefore, it remains unclear whether reflective reasoning will emerge during Markovian RL training, or why they are beneficial at test time. To remedy this, we recast reflective exploration within the Bayes-Adaptive RL framework, which explicitly optimizes the expected return under a posterior distribution over Markov decision processes. This Bayesian formulation inherently incentivizes both reward-maximizing exploitation and information-gathering exploration via belief updates. Our resulting algorithm, BARL, instructs the LLM to stitch and switch strategies based on the observed outcomes, offering principled guidance on when and how the model should reflectively explore. Empirical results on both synthetic and mathematical reasoning tasks demonstrate that BARL outperforms standard Markovian RL approaches at test time, achieving superior token efficiency with improved exploration effectiveness. Our code is available at https://github.com/shenao-zhang/BARL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13697v2">RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Reinforcement learning-based post-training of large language models (LLMs) has recently gained attention, particularly following the release of DeepSeek R1, which applied GRPO for fine-tuning. Amid the growing hype around improved reasoning abilities attributed to RL post-training, we critically examine the formulation and assumptions underlying these methods. We start by highlighting the popular structural assumptions made in modeling LLM training as a Markov Decision Process (MDP), and show how they lead to a degenerate MDP that doesn't quite need the RL/GRPO apparatus. The two critical structural assumptions include (1) making the MDP states be just a concatenation of the actions-with states becoming the context window and the actions becoming the tokens in LLMs and (2) splitting the reward of a state-action trajectory uniformly across the trajectory. Through a comprehensive analysis, we demonstrate that these simplifying assumptions make the approach effectively equivalent to an outcome-driven supervised learning. Our experiments on benchmarks including GSM8K and Countdown using Qwen-2.5 base models show that iterative supervised fine-tuning, incorporating both positive and negative samples, achieves performance comparable to GRPO-based training. We will also argue that the structural assumptions indirectly incentivize the RL to generate longer sequences of intermediate tokens-which in turn feeds into the narrative of "RL generating longer thinking traces." While RL may well be a very useful technique for improving the reasoning abilities of LLMs, our analysis shows that the simplistic structural assumptions made in modeling the underlying MDP render the popular LLM RL frameworks and their interpretations questionable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17117v2">From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20524v1">Towards Fully FP8 GEMM LLM Training at Scale</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Despite the significant potential of FP8 data formats for large language model (LLM) pre-training, their adoption has been limited due to challenges in maintaining stability at scale. Existing approaches often rely on suboptimal fine-grained FP8 kernels or fall back to higher-precision matrix multiplications (GEMMs) in sensitive components, such as attention projections, compromising potential throughput gains. We introduce a new class of LLM architectures that, for the first time, support FP8 computation for all GEMMs within transformer blocks during both forward and backward passes. This enables unprecedented throughput gains, particularly at scale, while matching the downstream performance of standard BF16 training. Our architecture design reduces large outlier activations, promoting stable long-term FP8 training. In addition, we identify key metrics to monitor low-precision training and predict potential future divergences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20521v1">Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 28 pages, 5 figures. Submitted for review to Information Fusion
    </div>
    <details class="paper-abstract">
      This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20501v1">Gatsby Without the 'E': Crafting Lipograms with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 7.5 pages
    </div>
    <details class="paper-abstract">
      Lipograms are a unique form of constrained writing where all occurrences of a particular letter are excluded from the text, typified by the novel Gadsby, which daringly avoids all usage of the letter 'e'. In this study, we explore the power of modern large language models (LLMs) by transforming the novel F. Scott Fitzgerald's The Great Gatsby into a fully 'e'-less text. We experimented with a range of techniques, from baseline methods like synonym replacement to sophisticated generative models enhanced with beam search and named entity analysis. We show that excluding up to 3.6% of the most common letters (up to the letter 'u') had minimal impact on the text's meaning, although translation fidelity rapidly and predictably decays with stronger lipogram constraints. Our work highlights the surprising flexibility of English under strict constraints, revealing just how adaptable and creative language can be.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20487v1">InFact: Informativeness Alignment for Improved LLM Factuality</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Factual completeness is a general term that captures how detailed and informative a factually correct text is. For instance, the factual sentence ``Barack Obama was born in the United States'' is factually correct, though less informative than the factual sentence ``Barack Obama was born in Honolulu, Hawaii, United States''. Despite the known fact that LLMs tend to hallucinate and generate factually incorrect text, they might also tend to choose to generate factual text that is indeed factually correct and yet less informative than other, more informative choices. In this work, we tackle this problem by proposing an informativeness alignment mechanism. This mechanism takes advantage of recent factual benchmarks to propose an informativeness alignment objective. This objective prioritizes answers that are both correct and informative. A key finding of our work is that when training a model to maximize this objective or optimize its preference, we can improve not just informativeness but also factuality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20451v1">Amulet: Putting Complex Multi-Turn Conversations on the Stand with LLM Juries</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Today, large language models are widely used as judges to evaluate responses from other language models. Hence, it is imperative to benchmark and improve these LLM-judges on real-world language model usage: a typical human-assistant conversation is lengthy, and shows significant diversity in topics, intents, and requirements across turns, e.g. social interactions, task requests, feedback. We present Amulet, a framework that leverages pertinent linguistic concepts of dialog-acts and maxims to improve the accuracy of LLM-judges on preference data with complex, multi-turn conversational context. Amulet presents valuable insights about (a) the communicative structures and intents present in the conversation (dialog acts), and (b) the satisfaction of conversational principles (maxims) by the preference responses, and uses them to make judgments. On four challenging datasets, Amulet shows that (a) humans frequently (60 to 70 percent of the time) change their intents from one turn of the conversation to the next, and (b) in 75 percent of instances, the preference responses can be differentiated via dialog acts and/or maxims, reiterating the latter's significance in judging such data. Amulet can be used either as a judge by applying the framework to a single LLM, or integrated into a jury with different LLM judges; our judges and juries show strong improvements on relevant baselines for all four datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20438v1">HAMburger: Accelerating LLM Inference via Token Smashing</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      The growing demand for efficient Large Language Model (LLM) inference requires a holistic optimization on algorithms, systems, and hardware. However, very few works have fundamentally changed the generation pattern: each token needs one forward pass and one KV cache. This can be sub-optimal because we found that LLMs are extremely capable of self-identifying the exact dose of information that a single KV cache can store, and many tokens can be generated confidently without global context. Based on this insight, we introduce HAMburger, a Hierarchically Auto-regressive Model that redefines resource allocation in LLMs by moving beyond uniform computation and storage per token during inference. Stacking a compositional embedder and a micro-step decoder in between a base LLM, HAMburger smashes multiple tokens into a single KV and generates several tokens per step. Additionally, HAMburger functions as a speculative decoding framework where it can blindly trust self-drafted tokens. As a result, HAMburger shifts the growth of KV cache and forward FLOPs from linear to sub-linear with respect to output length, and adjusts its inference speed based on query perplexity and output structure. Extensive evaluations show that HAMburger reduces the KV cache computation by up to 2$\times$ and achieves up to 2$\times$ TPS, while maintaining quality in both short- and long-context tasks. Our method explores an extremely challenging inference regime that requires both computation- and memory-efficiency with a hardware-agnostic design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20416v1">GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Fine-tuning for large language models (LLMs) typically requires substantial amounts of high-quality supervised data, which is both costly and labor-intensive to acquire. While synthetic data generation has emerged as a promising solution, existing approaches frequently suffer from factual inaccuracies, insufficient long-tail coverage, simplistic knowledge structures, and homogenized outputs. To address these challenges, we introduce GraphGen, a knowledge graph-guided framework designed for three key question-answering (QA) scenarios: atomic QA, aggregated QA, and multi-hop QA. It begins by constructing a fine-grained knowledge graph from the source text. It then identifies knowledge gaps in LLMs using the expected calibration error metric, prioritizing the generation of QA pairs that target high-value, long-tail knowledge. Furthermore, GraphGen incorporates multi-hop neighborhood sampling to capture complex relational information and employs style-controlled generation to diversify the resulting QA data. Experimental results on knowledge-intensive tasks under closed-book settings demonstrate that GraphGen outperforms conventional synthetic data methods, offering a more reliable and comprehensive solution to the data scarcity challenge in supervised fine-tuning. The code and data are publicly available at https://github.com/open-sciencelab/GraphGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20295v1">Self-reflective Uncertainties: Do LLMs Know Their Internal Answer Distribution?</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      To reveal when a large language model (LLM) is uncertain about a response, uncertainty quantification commonly produces percentage numbers along with the output. But is this all we can do? We argue that in the output space of LLMs, the space of strings, exist strings expressive enough to summarize the distribution over output strings the LLM deems possible. We lay a foundation for this new avenue of uncertainty explication and present SelfReflect, a theoretically-motivated metric to assess how faithfully a string summarizes an LLM's internal answer distribution. We show that SelfReflect is able to discriminate even subtle differences of candidate summary strings and that it aligns with human judgement, outperforming alternative metrics such as LLM judges and embedding comparisons. With SelfReflect, we investigate a number of self-summarization methods and find that even state-of-the-art reasoning models struggle to explicate their internal uncertainty. But we find that faithful summarizations can be generated by sampling and summarizing. Our metric enables future works towards this universal form of LLM uncertainties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20296v1">Reasoning LLMs are Wandering Solution Explorers</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 71 pages, 14 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive reasoning abilities through test-time computation (TTC) techniques such as chain-of-thought prompting and tree-based reasoning. However, we argue that current reasoning LLMs (RLLMs) lack the ability to systematically explore the solution space. This paper formalizes what constitutes systematic problem solving and identifies common failure modes that reveal reasoning LLMs to be wanderers rather than systematic explorers. Through qualitative and quantitative analysis across multiple state-of-the-art LLMs, we uncover persistent issues: invalid reasoning steps, redundant explorations, hallucinated or unfaithful conclusions, and so on. Our findings suggest that current models' performance can appear to be competent on simple tasks yet degrade sharply as complexity increases. Based on the findings, we advocate for new metrics and tools that evaluate not just final outputs but the structure of the reasoning process itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20073v2">RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) as interactive agents presents unique challenges including long-horizon decision making and interacting with stochastic environment feedback. While reinforcement learning (RL) has enabled progress in static tasks, multi-turn agent RL training remains underexplored. We propose StarPO (State-Thinking-Actions-Reward Policy Optimization), a general framework for trajectory-level agent RL, and introduce RAGEN, a modular system for training and evaluating LLM agents. Our study on four stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of Echo Trap where reward variance cliffs and gradient spikes; we address this with StarPO-S, a stabilized variant with trajectory filtering, critic incorporation, and gradient stabilization. Second, we find the shaping of RL rollouts would benefit from diverse initial states, medium interaction granularity and more frequent sampling. Third, we show that without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge through multi-turn RL and they may show shallow strategies or hallucinated thoughts. Code and environments are available at https://github.com/RAGEN-AI/RAGEN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20201v1">Reasoning Is Not All You Need: Examining LLMs for Multi-Turn Mental Health Conversations</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 33 pages, 5 figures, 30 tables
    </div>
    <details class="paper-abstract">
      Limited access to mental healthcare, extended wait times, and increasing capabilities of Large Language Models (LLMs) has led individuals to turn to LLMs for fulfilling their mental health needs. However, examining the multi-turn mental health conversation capabilities of LLMs remains under-explored. Existing evaluation frameworks typically focus on diagnostic accuracy and win-rates and often overlook alignment with patient-specific goals, values, and personalities required for meaningful conversations. To address this, we introduce MedAgent, a novel framework for synthetically generating realistic, multi-turn mental health sensemaking conversations and use it to create the Mental Health Sensemaking Dialogue (MHSD) dataset, comprising over 2,200 patient-LLM conversations. Additionally, we present MultiSenseEval, a holistic framework to evaluate the multi-turn conversation abilities of LLMs in healthcare settings using human-centric criteria. Our findings reveal that frontier reasoning models yield below-par performance for patient-centric communication and struggle at advanced diagnostic capabilities with average score of 31%. Additionally, we observed variation in model performance based on patient's persona and performance drop with increasing turns in the conversation. Our work provides a comprehensive synthetic data generation framework, a dataset and evaluation framework for assessing LLMs in multi-turn mental health conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20196v1">Temporal Sampling for Forgotten Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) is intended to improve their reasoning capabilities, yet we uncover a counterintuitive effect: models often forget how to solve problems they previously answered correctly during training. We term this phenomenon temporal forgetting and show that it is widespread across model sizes, fine-tuning methods (both Reinforcement Learning and Supervised Fine-Tuning), and multiple reasoning benchmarks. To address this gap, we introduce Temporal Sampling, a simple decoding strategy that draws outputs from multiple checkpoints along the training trajectory. This approach recovers forgotten solutions without retraining or ensembling, and leads to substantial improvements in reasoning performance, gains from 4 to 19 points in Pass@k and consistent gains in Majority@k across several benchmarks. We further extend our method to LoRA-adapted models, demonstrating that storing only adapter weights across checkpoints achieves similar benefits with minimal storage cost. By leveraging the temporal diversity inherent in training, Temporal Sampling offers a practical, compute-efficient way to surface hidden reasoning ability and rethink how we evaluate LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07965v4">SHARP: Unlocking Interactive Hallucination via Stance Transfer in Role-Playing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 28 pages, unfortunately accepted to findings with Meta 4.. Apologize for the reviewers and area chair who love our work, orz
    </div>
    <details class="paper-abstract">
      The advanced role-playing capabilities of Large Language Models (LLMs) have enabled rich interactive scenarios, yet existing research in social interactions neglects hallucination while struggling with poor generalizability and implicit character fidelity judgments. To bridge this gap, motivated by human behaviour, we introduce a generalizable and explicit paradigm for uncovering interactive patterns of LLMs across diverse worldviews. Specifically, we first define interactive hallucination through stance transfer, then construct SHARP, a benchmark built by extracting relations from commonsense knowledge graphs and utilizing LLMs' inherent hallucination properties to simulate multi-role interactions. Extensive experiments confirm our paradigm's effectiveness and stability, examine the factors that influence these metrics, and challenge conventional hallucination mitigation solutions. More broadly, our work reveals a fundamental limitation in popular post-training methods for role-playing LLMs: the tendency to obscure knowledge beneath style, resulting in monotonous yet human-like behaviors - interactive hallucination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17644v5">BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Serving systems for Large Language Models (LLMs) are often optimized to improve quality of service (QoS) and throughput. However, due to the lack of open-source LLM serving workloads, these systems are frequently evaluated under unrealistic workload assumptions. Consequently, performance may degrade when systems are deployed in real-world scenarios. This work presents BurstGPT, an LLM serving workload with 10.31 million traces from regional Azure OpenAI GPT services over 213 days. BurstGPT captures LLM serving characteristics from user, model and system perspectives: (1) User request concurrency: burstiness variations of requests in Azure OpenAI GPT services, revealing diversified concurrency patterns in different services and model types. (2) User conversation patterns: counts and intervals within conversations for service optimizations. (3) Model response lengths: auto-regressive serving processes of GPT models, showing statistical relations between requests and their responses. (4) System response failures: failures of conversation and API services, showing intensive resource needs and limited availability of LLM services in Azure. The details of the characteristics can serve multiple purposes in LLM serving optimizations, such as system evaluation and trace provisioning. In our demo evaluation with BurstGPT, frequent variations in BurstGPT reveal declines in efficiency, stability, or reliability in realistic LLM serving. We identify that the generalization of KV cache management, scheduling and disaggregation optimizations can be improved under realistic workload evaluations. BurstGPT is publicly available now at https://github.com/HPMLL/BurstGPT and is widely used to develop prototypes of LLM serving frameworks in the industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20162v1">Capability-Based Scaling Laws for LLM Red-Teaming</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      As large language models grow in capability and agency, identifying vulnerabilities through red-teaming becomes vital for safe deployment. However, traditional prompt-engineering approaches may prove ineffective once red-teaming turns into a weak-to-strong problem, where target models surpass red-teamers in capabilities. To study this shift, we frame red-teaming through the lens of the capability gap between attacker and target. We evaluate more than 500 attacker-target pairs using LLM-based jailbreak attacks that mimic human red-teamers across diverse families, sizes, and capability levels. Three strong trends emerge: (i) more capable models are better attackers, (ii) attack success drops sharply once the target's capability exceeds the attacker's, and (iii) attack success rates correlate with high performance on social science splits of the MMLU-Pro benchmark. From these trends, we derive a jailbreaking scaling law that predicts attack success for a fixed target based on attacker-target capability gap. These findings suggest that fixed-capability attackers (e.g., humans) may become ineffective against future models, increasingly capable open-source models amplify risks for existing systems, and model providers must accurately measure and control models' persuasive and manipulative abilities to limit their effectiveness as attackers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20161v1">Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Effective generalization in language models depends critically on the diversity of their training data. Yet existing diversity metrics often fall short of this goal, relying on surface-level heuristics that are decoupled from model behavior. This motivates us to ask: What kind of diversity in training data actually drives generalization in language models -- and how can we measure and amplify it? Through large-scale empirical analyses spanning over 300 training runs, carefully controlled for data scale and quality, we show that data diversity can be a strong predictor of generalization in LLM reasoning -- as measured by average model performance on unseen out-of-distribution benchmarks. We introduce G-Vendi, a metric that quantifies diversity via the entropy of model-induced gradients. Despite using a small off-the-shelf proxy model for gradients, G-Vendi consistently outperforms alternative measures, achieving strong correlation (Spearman's $\rho \approx 0.9$) with out-of-distribution (OOD) performance on both natural language inference (NLI) and math reasoning tasks. Building on this insight, we present Prismatic Synthesis, a framework for generating diverse synthetic data by targeting underrepresented regions in gradient space. Experimental results show that Prismatic Synthesis consistently improves model performance as we scale synthetic data -- not just on in-distribution test but across unseen, out-of-distribution benchmarks -- significantly outperforming state-of-the-art models that rely on 20 times larger data generator than ours. For example, PrismMath-7B, our model distilled from a 32B LLM, outperforms R1-Distill-Qwen-7B -- the same base model trained on proprietary data generated by 671B R1 -- on 6 out of 7 challenging benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07215v2">The CodeInverter Suite: Control-Flow and Data-Mapping Augmented Binary Decompilation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Binary decompilation plays a vital role in various cybersecurity and software engineering tasks. Recently, end-to-end decompilation methods powered by large language models (LLMs) have garnered significant attention due to their ability to generate highly readable source code with minimal human intervention. However, existing LLM-based approaches face several critical challenges, including limited capability in reconstructing code structure and logic, low accuracy in data recovery, concerns over data security and privacy, and high computational resource requirements. To address these issues, we develop the CodeInverter Suite, making three contributions: (1) the CodeInverter Workflow (CIW) is a novel prompt engineering workflow that incorporates control flow graphs (CFG) and explicit data mappings to improve LLM-based decompilation. (2) Using CIW on well-known source code datasets, we curate the CodeInverter Dataset (CID), a domain-specific dataset containing 8.69 million samples that contains CFGs and data mapping tables. (3) We train the CoderInverter Models (CIMs) on CID, generating two lightweight LLMs (with 1.3B and 6.7B parameters) intended for efficient inference in privacy-sensitive or resource-constrained environments. Extensive experiments on two benchmarks demonstrate that the CIW substantially enhances the performance of various LLMs across multiple metrics. Our CIM-6.7B can achieve state-of-the-art decompilation performance, outperforming existing LLMs even with over 100x more parameters in decompilation tasks, an average improvement of 11.03% in re-executability, 6.27% in edit similarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17773v3">Uncertainty Quantification for LLM-Based Survey Simulations</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 33 pages, 7 figures, 10 tables
    </div>
    <details class="paper-abstract">
      We investigate the use of large language models (LLMs) to simulate human responses to survey questions, and perform uncertainty quantification to gain reliable insights. Our approach converts imperfect LLM-simulated responses into confidence sets for population parameters of human responses, addressing the distribution shift between the simulated and real populations. A key innovation lies in determining the optimal number of simulated responses: too many produce overly narrow confidence sets with poor coverage, while too few yield excessively loose estimates. To resolve this, our method adaptively selects the simulation sample size, ensuring valid average-case coverage guarantees. It is broadly applicable to any LLM, irrespective of its fidelity, and any procedure for constructing confidence sets. Additionally, the selected sample size quantifies the degree of misalignment between the LLM and the target human population. We illustrate our method on real datasets and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13879v4">Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under Black-box Settings</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 22 pages, 8 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks yet still are vulnerable to external threats, particularly LLM Denial-of-Service (LLM-DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, existing studies predominantly focus on white-box attacks, leaving black-box scenarios underexplored. In this paper, we introduce Auto-Generation for LLM-DoS (AutoDoS) attack, an automated algorithm designed for black-box LLMs. AutoDoS constructs the DoS Attack Tree and expands the node coverage to achieve effectiveness under black-box conditions. By transferability-driven iterative optimization, AutoDoS could work across different models in one prompt. Furthermore, we reveal that embedding the Length Trojan allows AutoDoS to bypass existing defenses more effectively. Experimental results show that AutoDoS significantly amplifies service response latency by over 250$\times\uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our work provides a new perspective on LLM-DoS attacks and security defenses. Our code is available at https://github.com/shuita2333/AutoDoS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05668v3">JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Correct typos and update new experiment results. Accepted in ACL 2025. 25 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Jailbreak attacks aim to bypass the LLMs' safeguards. While researchers have proposed different jailbreak attacks in depth, they have done so in isolation -- either with unaligned settings or comparing a limited range of methods. To fill this gap, we present a large-scale evaluation of various jailbreak attacks. We collect 17 representative jailbreak attacks, summarize their features, and establish a novel jailbreak attack taxonomy. Then we conduct comprehensive measurement and ablation studies across nine aligned LLMs on 160 forbidden questions from 16 violation categories. Also, we test jailbreak attacks under eight advanced defenses. Based on our taxonomy and experiments, we identify some important patterns, such as heuristic-based attacks could achieve high attack success rates but are easy to mitigate by defenses, causing low practicality. Our study offers valuable insights for future research on jailbreak attacks and defenses. We hope our work could help the community avoid incremental work and serve as an effective benchmark tool for practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20354v1">Rethinking Text-based Protein Understanding: Retrieval or LLM?</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      In recent years, protein-text models have gained significant attention for their potential in protein generation and understanding. Current approaches focus on integrating protein-related knowledge into large language models through continued pretraining and multi-modal alignment, enabling simultaneous comprehension of textual descriptions and protein sequences. Through a thorough analysis of existing model architectures and text-based protein understanding benchmarks, we identify significant data leakage issues present in current benchmarks. Moreover, conventional metrics derived from natural language processing fail to accurately assess the model's performance in this domain. To address these limitations, we reorganize existing datasets and introduce a novel evaluation framework based on biological entities. Motivated by our observation, we propose a retrieval-enhanced method, which significantly outperforms fine-tuned LLMs for protein-to-text generation and shows accuracy and efficiency in training-free scenarios. Our code and data can be seen at https://github.com/IDEA-XL/RAPM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20155v1">Pangu Light: Weight Re-Initialization for Pruning and Accelerating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver state-of-the-art capabilities across numerous tasks, but their immense size and inference costs pose significant computational challenges for practical deployment. While structured pruning offers a promising avenue for model compression, existing methods often struggle with the detrimental effects of aggressive, simultaneous width and depth reductions, leading to substantial performance degradation. This paper argues that a critical, often overlooked, aspect in making such aggressive joint pruning viable is the strategic re-initialization and adjustment of remaining weights to improve the model post-pruning training accuracies. We introduce Pangu Light, a framework for LLM acceleration centered around structured pruning coupled with novel weight re-initialization techniques designed to address this ``missing piece''. Our framework systematically targets multiple axes, including model width, depth, attention heads, and RMSNorm, with its effectiveness rooted in novel re-initialization methods like Cross-Layer Attention Pruning (CLAP) and Stabilized LayerNorm Pruning (SLNP) that mitigate performance drops by providing the network a better training starting point. Further enhancing efficiency, Pangu Light incorporates specialized optimizations such as absorbing Post-RMSNorm computations and tailors its strategies to Ascend NPU characteristics. The Pangu Light models consistently exhibit a superior accuracy-efficiency trade-off, outperforming prominent baseline pruning methods like Nemotron and established LLMs like Qwen3 series. For instance, on Ascend NPUs, Pangu Light-32B's 81.6 average score and 2585 tokens/s throughput exceed Qwen3-32B's 80.9 average score and 2225 tokens/s.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15873v2">Abstractions-of-Thought: Intermediate Representations for LLM Reasoning in Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive proficiency on logic and programming tasks, often rivaling expert-level performance. However, generating functionally correct hardware description language (HDL) code from natural language specifications remains challenging, primarily in data-scarce domains. Therefore, we present Abstractions-of-Thought (AoT) - a training-free, inference-only prompting framework to mitigate misinterpretations and reasoning pitfalls of LLMs through a series of task-based abstractions within the prompting procedure, assisting in the transition from high-level to low-level representations of hardware. Furthermore, AoT consists of the following stages: (1) an LLM-based classification of hardware design patterns, (2) a structured intermediate representation (IR) to separate functional decomposition from code syntax, and (3) a line-by-line pseudocode solution enabling a more direct mapping to the final Verilog implementation. Experimental results on the VerilogEval benchmark depict that AoT demonstrates improvements in functionality when applied to large non-reasoning models (such as GPT-4o, outperforming all baseline techniques (including 1-shot, Chain-of-Thought, and Tree-of-Thought) while significantly reducing the generated tokens by 1.8-5.2x compared to popular Tree-of-Thought prompting.
    </details>
</div>
