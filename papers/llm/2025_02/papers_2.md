# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18449v1">SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The recent DeepSeek-R1 release has demonstrated the immense potential of reinforcement learning (RL) in enhancing the general reasoning capabilities of large language models (LLMs). While DeepSeek-R1 and other follow-up work primarily focus on applying RL to competitive coding and math problems, this paper introduces SWE-RL, the first approach to scale RL-based LLM reasoning for real-world software engineering. Leveraging a lightweight rule-based reward (e.g., the similarity score between ground-truth and LLM-generated solutions), SWE-RL enables LLMs to autonomously recover a developer's reasoning processes and solutions by learning from extensive open-source software evolution data -- the record of a software's entire lifecycle, including its code snapshots, code changes, and events such as issues and pull requests. Trained on top of Llama 3, our resulting reasoning model, Llama3-SWE-RL-70B, achieves a 41.0% solve rate on SWE-bench Verified -- a human-verified collection of real-world GitHub issues. To our knowledge, this is the best performance reported for medium-sized (<100B) LLMs to date, even comparable to leading proprietary LLMs like GPT-4o. Surprisingly, despite performing RL solely on software evolution data, Llama3-SWE-RL has even emerged with generalized reasoning skills. For example, it shows improved results on five out-of-domain tasks, namely, function coding, library use, code reasoning, mathematics, and general language understanding, whereas a supervised-finetuning baseline even leads to performance degradation on average. Overall, SWE-RL opens up a new direction to improve the reasoning capabilities of LLMs through reinforcement learning on massive software engineering data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11709v3">Towards Detecting Prompt Knowledge Gaps for Improved LLM-guided Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential in software development, especially for issue resolution. However, despite their widespread use, significant challenges persist in the quality of LLM responses to issue resolution queries. LLM interactions often yield incorrect, incomplete, or ambiguous information, largely due to knowledge gaps in prompt design, which can lead to unproductive exchanges and reduced developer productivity. In this paper, we analyze 433 developer-ChatGPT conversations within GitHub issue threads to examine the impact of prompt knowledge gaps and conversation styles on issue resolution. We identify four main knowledge gaps in developer prompts: Missing Context, Missing Specifications, Multiple Context, and Unclear Instructions. Assuming that conversations within closed issues contributed to successful resolutions while those in open issues did not, we find that ineffective conversations contain knowledge gaps in 44.6% of prompts, compared to only 12.6% in effective ones. Additionally, we observe seven distinct conversational styles, with Directive Prompting, Chain of Thought, and Responsive Feedback being the most prevalent. We find that knowledge gaps are present in all styles of conversations, with Missing Context being the most repeated challenge developers face in issue-resolution conversations. Based on our analysis, we identify key textual and code-related heuristics (Specificity, Contextual Richness, and Clarity) that are associated with successful issue closure and help assess prompt quality. These heuristics lay the foundation for an automated tool that can dynamically flag unclear prompts and suggest structured improvements. To test feasibility, we developed a lightweight browser extension prototype for detecting prompt gaps, that can be easily adapted to other tools within developer workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10563v2">Accelerating Unbiased LLM Evaluation via Synthetic Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      When developing new large language models (LLMs), a key step is evaluating their final performance, often by computing the win-rate against a reference model based on external feedback. Human feedback is the gold standard, particularly for capturing nuanced qualities like coherence, readability, and alignment with human expectations. However, human evaluations are costly -- even for large tech companies -- and when conducted with active users, they may negatively impact user experience. A promising alternative is synthetic feedback, where evaluations are conducted by other large language models, including reward models. While this eliminates the need for costly human annotations, it introduces biases that may distort the evaluation process. In this work, we propose a statistically principled framework that integrates human and synthetic feedback to reduce reliance on human annotations while maintaining unbiased win-rate calculations. Our experiments demonstrate a reduction in human annotations by up to 12.2% with an off-the-shelf synthetic evaluator and up to 24.8% with a finetuned variant. Apart from being generalizable, scalable, and free of hyper-parameter tuning, our method offers predictable annotation savings, which can be estimated based on data-dependent characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18414v1">GLEAN: Generalized Category Discovery with Diverse and Quality-Enhanced LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Generalized Category Discovery (GCD) is a practical and challenging open-world task that aims to recognize both known and novel categories in unlabeled data using limited labeled data from known categories. Due to the lack of supervision, previous GCD methods face significant challenges, such as difficulty in rectifying errors for confusing instances, and inability to effectively uncover and leverage the semantic meanings of discovered clusters. Therefore, additional annotations are usually required for real-world applicability. However, human annotation is extremely costly and inefficient. To address these issues, we propose GLEAN, a unified framework for generalized category discovery that actively learns from diverse and quality-enhanced LLM feedback. Our approach leverages three different types of LLM feedback to: (1) improve instance-level contrastive features, (2) generate category descriptions, and (3) align uncertain instances with LLM-selected category descriptions. Extensive experiments demonstrate the superior performance of \MethodName over state-of-the-art models across diverse datasets, metrics, and supervision settings. Our code is available at https://github.com/amazon-science/Glean.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18413v1">When Benchmarks Talk: Re-Evaluating Code LLMs with Interactive Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Programming is a fundamentally interactive process, yet coding assistants are often evaluated using static benchmarks that fail to measure how well models collaborate with users. We introduce an interactive evaluation pipeline to examine how LLMs incorporate different types of feedback in a collaborative setting. Specifically, we perturb static coding benchmarks so that the code model must interact with a simulated user to retrieve key information about the problem. We find that interaction significantly affects model performance, as the relative rankings of 10 models across 3 datasets often vary between static and interactive settings, despite models being fairly robust to feedback that contains errors. We also observe that even when different feedback types are equally effective with respect to performance, they can impact model behaviors such as (1) how models respond to higher- vs. lower-quality feedback and (2) whether models prioritize aesthetic vs. functional edits. Our work aims to "re-evaluate" model coding capabilities through an interactive lens toward bridging the gap between existing evaluations and real-world usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18389v1">Monte Carlo Temperature: a robust sampling strategy for LLM's uncertainty quantification methods</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) in Large Language Models (LLMs) is essential for their safe and reliable deployment, particularly in critical applications where incorrect outputs can have serious consequences. Current UQ methods typically rely on querying the model multiple times using non-zero temperature sampling to generate diverse outputs for uncertainty estimation. However, the impact of selecting a given temperature parameter is understudied, and our analysis reveals that temperature plays a fundamental role in the quality of uncertainty estimates. The conventional approach of identifying optimal temperature values requires expensive hyperparameter optimization (HPO) that must be repeated for each new model-dataset combination. We propose Monte Carlo Temperature (MCT), a robust sampling strategy that eliminates the need for temperature calibration. Our analysis reveals that: 1) MCT provides more robust uncertainty estimates across a wide range of temperatures, 2) MCT improves the performance of UQ methods by replacing fixed-temperature strategies that do not rely on HPO, and 3) MCT achieves statistical parity with oracle temperatures, which represent the ideal outcome of a well-tuned but computationally expensive HPO process. These findings demonstrate that effective UQ can be achieved without the computational burden of temperature parameter calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18387v1">How Far are LLMs from Real Search? A Comprehensive Study on Efficiency, Completeness, and Inherent Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 31 pages, 9 figures, 18 tables
    </div>
    <details class="paper-abstract">
      Search plays a fundamental role in problem-solving across various domains, with most real-world decision-making problems being solvable through systematic search. Drawing inspiration from recent discussions on search and learning, we systematically explore the complementary relationship between search and Large Language Models (LLMs) from three perspectives. First, we analyze how learning can enhance search efficiency and propose Search via Learning (SeaL), a framework that leverages LLMs for effective and efficient search. Second, we further extend SeaL to SeaL-C to ensure rigorous completeness during search. Our evaluation across three real-world planning tasks demonstrates that SeaL achieves near-perfect accuracy while reducing search spaces by up to 99.1% compared to traditional approaches. Finally, we explore how far LLMs are from real search by investigating whether they can develop search capabilities independently. Our analysis reveals that while current LLMs struggle with efficient search in complex problems, incorporating systematic search strategies significantly enhances their problem-solving capabilities. These findings not only validate the effectiveness of our approach but also highlight the need for improving LLMs' search abilities for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17962v5">Crafting Customisable Characters with LLMs: Introducing SimsChat, a Persona-Driven Role-Playing Agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable ability to comprehend instructions and generate human-like text, enabling sophisticated agent simulation beyond basic behavior replication. However, the potential for creating freely customisable characters remains underexplored. We introduce the Customisable Conversation Agent Framework, which employs LLMs to simulate real-world characters through personalised characteristic feature injection, enabling diverse character creation according to user preferences. We propose the SimsConv dataset, comprising 68 customised characters and 13,971 multi-turn role-playing dialogues across 1,360 real-world scenes. Characters are initially customised using pre-defined elements (career, aspiration, traits, skills), then expanded through personal and social profiles. Building on this, we present SimsChat, a freely customisable role-playing agent incorporating various realistic settings and topic-specified character interactions. Experimental results on both SimsConv and WikiRoleEval datasets demonstrate SimsChat's superior performance in maintaining character consistency, knowledge accuracy, and appropriate question rejection compared to existing models. Our framework provides valuable insights for developing more accurate and customisable human simulacra. Our data and code are publicly available at https://github.com/Bernard-Yang/SimsChat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03537v2">Ward: Provable RAG Dataset Inference via LLM Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      RAG enables LLMs to easily incorporate external data, raising concerns for data owners regarding unauthorized usage of their content. The challenge of detecting such unauthorized usage remains underexplored, with datasets and methods from adjacent fields being ill-suited for its study. We take several steps to bridge this gap. First, we formalize this problem as (black-box) RAG Dataset Inference (RAG-DI). We then introduce a novel dataset designed for realistic benchmarking of RAG-DI methods, alongside a set of baselines. Finally, we propose Ward, a method for RAG-DI based on LLM watermarks that equips data owners with rigorous statistical guarantees regarding their dataset's misuse in RAG corpora. Ward consistently outperforms all baselines, achieving higher accuracy, superior query efficiency and robustness. Our work provides a foundation for future studies of RAG-DI and highlights LLM watermarks as a promising approach to this problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07267v3">Transforming Role Classification in Scientific Teams Using LLMs and Advanced Predictive Analytics</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Accepted by Quantitative Science Studies (QSS)
    </div>
    <details class="paper-abstract">
      Scientific team dynamics are critical in determining the nature and impact of research outputs. However, existing methods for classifying author roles based on self-reports and clustering lack comprehensive contextual analysis of contributions. Thus, we present a transformative approach to classifying author roles in scientific teams using advanced large language models (LLMs), which offers a more refined analysis compared to traditional clustering methods. Specifically, we seek to complement and enhance these traditional methods by utilizing open source and proprietary LLMs, such as GPT-4, Llama3 70B, Llama2 70B, and Mistral 7x8B, for role classification. Utilizing few-shot prompting, we categorize author roles and demonstrate that GPT-4 outperforms other models across multiple categories, surpassing traditional approaches such as XGBoost and BERT. Our methodology also includes building a predictive deep learning model using 10 features. By training this model on a dataset derived from the OpenAlex database, which provides detailed metadata on academic publications -- such as author-publication history, author affiliation, research topics, and citation counts -- we achieve an F1 score of 0.76, demonstrating robust classification of author roles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18308v1">RefuteBench 2.0 -- Agentic Benchmark for Dynamic Evaluation of LLM Responses to Refutation Instruction</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Work on progess
    </div>
    <details class="paper-abstract">
      In the multi-turn interaction schema, large language models (LLMs) can leverage user feedback to enhance the quality and relevance of their responses. However, evaluating an LLM's ability to incorporate user refutation feedback is crucial yet challenging. In this study, we introduce RefuteBench 2.0, which significantly extends the original RefuteBench by incorporating LLM agents as refuters and evaluators, which allows for flexible and comprehensive assessment. We design both transient and persistent refutation instructions with different validity periods. Meta-evaluation shows that the LLM-based refuter could generate more human-like refutations and the evaluators could assign scores with high correlation with humans. Experimental results of various LLMs show that current models could effectively satisfy the refutation but fail to memorize the refutation information. Interestingly, we also observe that the performance of the initial task decreases as the refutations increase. Analysis of the attention scores further shows a potential weakness of current LLMs: they struggle to retain and correctly use previous information during long context dialogues. https://github.com/ElliottYan/RefuteBench-2.0
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13834v2">Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Published as a conference paper at ICLR 2025. Code is available at https://github.com/Lizn-zn/NeqLIPS/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~1). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04665v2">LLM-based MOFs Synthesis Condition Extraction using Few-Shot Demonstrations</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The extraction of Metal-Organic Frameworks (MOFs) synthesis route from literature has been crucial for the logical MOFs design with desirable functionality. The recent advent of large language models (LLMs) provides disruptively new solution to this long-standing problem. While the latest researches mostly stick to primitive zero-shot LLMs lacking specialized material knowledge, we introduce in this work the few-shot LLM in-context learning paradigm. First, a human-AI interactive data curation approach is proposed to secure high-quality demonstrations. Second, an information retrieval algorithm is applied to pick and quantify few-shot demonstrations for each extraction. Over three datasets randomly sampled from nearly 90,000 well-defined MOFs, we conduct triple evaluations to validate our method. The synthesis extraction, structure inference, and material design performance of the proposed few-shot LLMs all significantly outplay zero-shot LLM and baseline methods. The lab-synthesized material guided by LLM surpasses 91.1% high-quality MOFs of the same class reported in the literature, on the key physical property of specific surface area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18282v1">Better Aligned with Survey Respondents or Training Data? Unveiling Political Leanings of LLMs on U.S. Supreme Court Cases</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 under review
    </div>
    <details class="paper-abstract">
      The increased adoption of Large Language Models (LLMs) and their potential to shape public opinion have sparked interest in assessing these models' political leanings. Building on previous research that compared LLMs and human opinions and observed political bias in system responses, we take a step further to investigate the underlying causes of such biases by empirically examining how the values and biases embedded in training corpora shape model outputs. Specifically, we propose a method to quantitatively evaluate political leanings embedded in the large pretraining corpora. Subsequently we investigate to whom are the LLMs' political leanings more aligned with, their pretrainig corpora or the surveyed human opinions. As a case study, we focus on probing the political leanings of LLMs in 32 U.S. Supreme Court cases, addressing contentious topics such as abortion and voting rights. Our findings reveal that LLMs strongly reflect the political leanings in their training data, and no strong correlation is observed with their alignment to human opinions as expressed in surveys. These results underscore the importance of responsible curation of training data and the need for robust evaluation metrics to ensure LLMs' alignment with human-centered values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11853v3">Chat Bankman-Fried: an Exploration of LLM Alignment in Finance</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Advancements in large language models (LLMs) have renewed concerns about AI alignment - the consistency between human and AI goals and values. As various jurisdictions enact legislation on AI safety, the concept of alignment must be defined and measured across different domains. This paper proposes an experimental framework to assess whether LLMs adhere to ethical and legal standards in the relatively unexplored context of finance. We prompt twelve LLMs to impersonate the CEO of a financial institution and test their willingness to misuse customer assets to repay outstanding corporate debt. Beginning with a baseline configuration, we adjust preferences, incentives and constraints, analyzing the impact of each adjustment with logistic regression. Our findings reveal significant heterogeneity in the baseline propensity for unethical behavior of LLMs. Factors such as risk aversion, profit expectations, and regulatory environment consistently influence misalignment in ways predicted by economic theory, although the magnitude of these effects varies across LLMs. This paper highlights both the benefits and limitations of simulation-based, ex post safety testing. While it can inform financial authorities and institutions aiming to ensure LLM safety, there is a clear trade-off between generality and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16022v2">Enhancing LLMs for Identifying and Prioritizing Important Medical Jargons from Electronic Health Record Notes Utilizing Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 21pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      OpenNotes enables patients to access EHR notes, but medical jargon can hinder comprehension. To improve understanding, we evaluated closed- and open-source LLMs for extracting and prioritizing key medical terms using prompting, fine-tuning, and data augmentation. We assessed LLMs on 106 expert-annotated EHR notes, experimenting with (i) general vs. structured prompts, (ii) zero-shot vs. few-shot prompting, (iii) fine-tuning, and (iv) data augmentation. To enhance open-source models in low-resource settings, we used ChatGPT for data augmentation and applied ranking techniques. We incrementally increased the augmented dataset size (10 to 10,000) and conducted 5-fold cross-validation, reporting F1 score and Mean Reciprocal Rank (MRR). Our result show that fine-tuning and data augmentation improved performance over other strategies. GPT-4 Turbo achieved the highest F1 (0.433), while Mistral7B with data augmentation had the highest MRR (0.746). Open-source models, when fine-tuned or augmented, outperformed closed-source models. Notably, the best F1 and MRR scores did not always align. Few-shot prompting outperformed zero-shot in vanilla models, and structured prompts yielded different preferences across models. Fine-tuning improved zero-shot performance but sometimes degraded few-shot performance. Data augmentation performed comparably or better than other methods. Our evaluation highlights the effectiveness of prompting, fine-tuning, and data augmentation in improving model performance for medical jargon extraction in low-resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18210v1">From ChatGPT to DeepSeek: Can LLMs Simulate Humanity?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Simulation powered by Large Language Models (LLMs) has become a promising method for exploring complex human social behaviors. However, the application of LLMs in simulations presents significant challenges, particularly regarding their capacity to accurately replicate the complexities of human behaviors and societal dynamics, as evidenced by recent studies highlighting discrepancies between simulated and real-world interactions. We rethink LLM-based simulations by emphasizing both their limitations and the necessities for advancing LLM simulations. By critically examining these challenges, we aim to offer actionable insights and strategies for enhancing the applicability of LLM simulations in human society in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18209v1">LAG: LLM agents for Leaderboard Auto Generation on Demanding</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      This paper introduces Leaderboard Auto Generation (LAG), a novel and well-organized framework for automatic generation of leaderboards on a given research topic in rapidly evolving fields like Artificial Intelligence (AI). Faced with a large number of AI papers updated daily, it becomes difficult for researchers to track every paper's proposed methods, experimental results, and settings, prompting the need for efficient automatic leaderboard construction. While large language models (LLMs) offer promise in automating this process, challenges such as multi-document summarization, leaderboard generation, and experiment fair comparison still remain under exploration. LAG solves these challenges through a systematic approach that involves the paper collection, experiment results extraction and integration, leaderboard generation, and quality evaluation. Our contributions include a comprehensive solution to the leaderboard construction problem, a reliable evaluation method, and experimental results showing the high quality of leaderboards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18201v1">Intersubjective Model of AI-mediated Communication: Augmenting Human-Human Text Chat through LLM-based Adaptive Agent Pair</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The growing prevalence of Large Language Models (LLMs) is reshaping online text-based communication; a transformation that is extensively studied as AI-mediated communication. However, much of the existing research remains bound by traditional communication models, where messages are created and transmitted directly between humans despite LLMs being able to play a more active role in transforming messages. In this work, we propose the Intersubjective Model of AI-mediated Communication, an alternative communication model that leverages LLM-based adaptive agents to augment human-human communication. Unlike traditional communication models that focus on the accurate transmission of information, the Intersubjective Model allows for communication to be designed in an adaptive and customizable way to create alternative interactions by dynamically shaping messages in real time and facilitating shared understanding between the human participants. In this paper, we have developed a prototype text chat system based on the Intersubjective Model to describe the potential of this model, as well as the design space it affords.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10245v4">From Text to Emoji: How PEFT-Driven Personality Manipulation Unleashes the Emoji Potential in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Findings paper of NAACL 2025 and NeurIPS 2024 Workshop on Behavioral Machine Learning
    </div>
    <details class="paper-abstract">
      The manipulation of the personality traits of large language models (LLMs) has emerged as a key area of research. Methods like prompt-based In-Context Knowledge Editing (IKE) and gradient-based Model Editor Networks (MEND) have been explored but show irregularity and variability; IKE depends on the prompt, leading to variability and sensitivity, while MEND yields inconsistent and gibberish outputs. To address this, we employed Opinion QA Based Parameter-Efficient Fine-Tuning (PEFT), specifically Quantized Low-Rank Adaptation (QLoRA), to manipulate the Big Five personality traits: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. After PEFT, models such as Mistral-7B-Instruct and LLaMA-2-7B-chat showed a latent behaviour by generating emojis for certain traits, despite no emojis being present in the PEFT data. For instance, LLaMA-2-7B-chat generated emojis in 99.5\% of extraversion-related test instances, while Mistral-7B-Instruct did so in 92.5\% of openness-related test instances. ICL Explainability analysis indicated that the LLMs used emojis intentionally to express these traits. Mechanistic Interpretability analysis showed that this latent behaviour of LLMs could be traced to specific neurons that became activated or amplified after PEFT. This paper provides a number of novel contributions. First, introducing an Opinion QA dataset for PEFT-driven personality manipulation; second, developing metric models to benchmark LLM personality traits; third, demonstrating PEFT's superiority over IKE in personality manipulation; and finally, analysing and validating emoji usage through explainability methods such as Mechanistic Interpretability and In-context learning Explainability methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18179v1">Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study delves into the sub-problems within these core challenges, such as input representation, chunking, prompting, and selection of LLMs and multimodal models. It examines the outcomes of different design choices through a new layout-aware IE test suite, benchmarking against the state-of-art (SoA) model LayoutLMv3. The results show that the configuration from one-factor-at-a-time (OFAT) trial achieves near-optimal results with 14.1 points F1-score gain from the baseline model, while full factorial exploration yields only a slightly higher 15.1 points gain at around 36x greater token usage. We demonstrate that well-configured general-purpose LLMs can match the performance of specialized models, providing a cost-effective alternative. Our test-suite is freely available at https://github.com/gayecolakoglu/LayIE-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18156v1">Can LLMs Explain Themselves Counterfactually?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Explanations are an important tool for gaining insights into the behavior of ML models, calibrating user trust and ensuring regulatory compliance. Past few years have seen a flurry of post-hoc methods for generating model explanations, many of which involve computing model gradients or solving specially designed optimization problems. However, owing to the remarkable reasoning abilities of Large Language Model (LLMs), self-explanation, that is, prompting the model to explain its outputs has recently emerged as a new paradigm. In this work, we study a specific type of self-explanations, self-generated counterfactual explanations (SCEs). We design tests for measuring the efficacy of LLMs in generating SCEs. Analysis over various LLM families, model sizes, temperature settings, and datasets reveals that LLMs sometimes struggle to generate SCEs. Even when they do, their prediction often does not agree with their own counterfactual reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13603v2">Efficient Safety Retrofitting Against Jailbreaking for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) is an efficient alignment technique that steers LLMs towards preferable outputs by training on preference data, bypassing the need for explicit reward models. Its simplicity enables easy adaptation to various domains and safety requirements. This paper examines DPO's effectiveness in model safety against jailbreaking attacks while minimizing data requirements and training costs. We introduce Egida, a dataset expanded from multiple sources, which includes 27 different safety topics and 18 different attack styles, complemented with synthetic and human labels. This data is used to boost the safety of state-of-the-art LLMs (Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack styles. In addition to safety evaluations, we assess their post-alignment performance degradation in general purpose tasks, and their tendency to over refusal. Following the proposed methodology, trained models reduce their Attack Success Rate by 10%-30%, using small training efforts (2,000 samples) with low computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned models generalize to unseen topics and attack styles, with the most successful attack style reaching a success rate around 5%. Size and family are found to strongly influence model malleability towards safety, pointing at the importance of pre-training choices. To validate our findings, a large independent assessment of human preference agreement with Llama-Guard-3-8B is conducted by the authors and the associated dataset Egida-HSafe is released. Overall, this study illustrates how affordable and accessible it is to enhance LLM safety using DPO while outlining its current limitations. All datasets and models are released to enable reproducibility and further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00339v2">Rethinking Layer Removal: A Hybrid Pruning Framework Combining Layer Removal and Singular Value Selection for Efficient LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Layer removal is an effective technique for compressing large language models (LLMs) by reducing redundancy and improving inference efficiency. However, indiscriminate pruning disrupts representation stability, leading to performance degradation. We propose GRASP (Gradient-based Retention of Adaptive Singular Parameters), which preserves representation-critical singular values to mitigate these effects. Unlike direct layer removal, GRASP leverages gradient-based attribution on a syntax- and semantics-rich dataset to guide the selection of representation-critical singular values. By selectively applying singular value decomposition (SVD) to affected layers, GRASP achieves efficient compression while maintaining representation stability with minimal overhead. Experiments across multiple LLMs show that GRASP consistently outperforms existing compression methods in perplexity and downstream task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18125v1">HyperG: Hypergraph-Enhanced LLMs for Structured Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Given that substantial amounts of domain-specific knowledge are stored in structured formats, such as web data organized through HTML, Large Language Models (LLMs) are expected to fully comprehend this structured information to broaden their applications in various real-world downstream tasks. Current approaches for applying LLMs to structured data fall into two main categories: serialization-based and operation-based methods. Both approaches, whether relying on serialization or using SQL-like operations as an intermediary, encounter difficulties in fully capturing structural relationships and effectively handling sparse data. To address these unique characteristics of structured data, we propose HyperG, a hypergraph-based generation framework aimed at enhancing LLMs' ability to process structured knowledge. Specifically, HyperG first augment sparse data with contextual information, leveraging the generative power of LLMs, and incorporate a prompt-attentive hypergraph learning (PHL) network to encode both the augmented information and the intricate structural relationships within the data. To validate the effectiveness and generalization of HyperG, we conduct extensive experiments across two different downstream tasks requiring structured knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18116v1">Bayesian Optimization for Controlled Image Editing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 8 figures
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18080v1">Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Recent studies have shown that making a model spend more time thinking through longer Chain of Thoughts (CoTs) enables it to gain significant improvements in complex reasoning tasks. While current researches continue to explore the benefits of increasing test-time compute by extending the CoT lengths of Large Language Models (LLMs), we are concerned about a potential issue hidden behind the current pursuit of test-time scaling: Would excessively scaling the CoT length actually bring adverse effects to a model's reasoning performance? Our explorations on mathematical reasoning tasks reveal an unexpected finding that scaling with longer CoTs can indeed impair the reasoning performance of LLMs in certain domains. Moreover, we discover that there exists an optimal scaled length distribution that differs across different domains. Based on these insights, we propose a Thinking-Optimal Scaling strategy. Our method first uses a small set of seed data with varying response length distributions to teach the model to adopt different reasoning efforts for deep thinking. Then, the model selects its shortest correct response under different reasoning efforts on additional problems for self-improvement. Our self-improved models built upon Qwen2.5-32B-Instruct outperform other distillation-based 32B o1-like models across various math benchmarks, and achieve performance on par with QwQ-32B-Preview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14502v2">How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) on many tasks is greatly limited by the knowledge learned during pre-training and stored in the model's parameters. Low-rank adaptation (LoRA) is a popular and efficient training technique for updating or domain-specific adaptation of LLMs. In this study, we investigate how new facts can be incorporated into the LLM using LoRA without compromising the previously learned knowledge. We fine-tuned Llama-3.1-8B-instruct using LoRA with varying amounts of new knowledge. Our experiments have shown that the best results are obtained when the training data contains a mixture of known and new facts. However, this approach is still potentially harmful because the model's performance on external question-answering benchmarks declines after such fine-tuning. When the training data is biased towards certain entities, the model tends to regress to few overrepresented answers. In addition, we found that the model becomes more confident and refuses to provide an answer in only few cases. These findings highlight the potential pitfalls of LoRA-based LLM updates and underscore the importance of training data composition and tuning parameters to balance new knowledge integration and general model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18697v2">How Good Are LLMs for Literary Translation, Really? Literary Translation Evaluation with Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 NAACL Camera-Ready version
    </div>
    <details class="paper-abstract">
      Recent research has focused on literary machine translation (MT) as a new challenge in MT. However, the evaluation of literary MT remains an open problem. We contribute to this ongoing discussion by introducing LITEVAL-CORPUS, a paragraph-level parallel corpus containing verified human translations and outputs from 9 MT systems, which totals over 2k translations and 13k evaluated sentences across four language pairs, costing 4.5k C. This corpus enables us to (i) examine the consistency and adequacy of human evaluation schemes with various degrees of complexity, (ii) compare evaluations by students and professionals, assess the effectiveness of (iii) LLM-based metrics and (iv) LLMs themselves. Our findings indicate that the adequacy of human evaluation is controlled by two factors: the complexity of the evaluation scheme (more complex is less adequate) and the expertise of evaluators (higher expertise yields more adequate evaluations). For instance, MQM (Multidimensional Quality Metrics), a complex scheme and the de facto standard for non-literary human MT evaluation, is largely inadequate for literary translation evaluation: with student evaluators, nearly 60% of human translations are misjudged as indistinguishable or inferior to machine translations. In contrast, BWS (BEST-WORST SCALING), a much simpler scheme, identifies human translations at a rate of 80-100%. Automatic metrics fare dramatically worse, with rates of at most 20%. Our overall evaluation indicates that published human translations consistently outperform LLM translations, where even the most recent LLMs tend to produce considerably more literal and less diverse translations compared to humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v1">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 9 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference", and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13178v3">SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 23 pages, 17 tables, 8 figures
    </div>
    <details class="paper-abstract">
      With the integration of large language models (LLMs), embodied agents have strong capabilities to process the scene information and plan complicated instructions in natural language, paving the way for the potential deployment of embodied robots. However, a foreseeable issue is that those embodied agents can also flawlessly execute some hazardous tasks, potentially causing damages in the real world. To study this issue, we present SafeAgentBench-a new benchmark for safety-aware task planning of embodied LLM agents. SafeAgentBench includes: (1) a new dataset with 750 tasks, covering 10 potential hazards and 3 task types; (2) SafeAgentEnv, a universal embodied environment with a low-level controller, supporting multi-agent execution with 17 high-level actions for 8 state-of-the-art baselines; and (3) reliable evaluation methods from both execution and semantic perspectives. Experimental results show that, although agents based on different design frameworks exhibit substantial differences in task success rates, their overall safety awareness remains weak. The most safety-conscious baseline achieves only a 10\% rejection rate for detailed hazardous tasks. Moreover, simply replacing the LLM driving the agent does not lead to notable improvements in safety awareness. More details and code are available at https://github.com/shengyin1224/SafeAgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19487v4">HealthQ: Unveiling Questioning Capabilities of LLM Chains in Healthcare Conversations</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Effective patient care in digital healthcare requires large language models (LLMs) that not only answer questions but also actively gather critical information through well-crafted inquiries. This paper introduces HealthQ, a novel framework for evaluating the questioning capabilities of LLM healthcare chains. By implementing advanced LLM chains, including Retrieval-Augmented Generation (RAG), Chain of Thought (CoT), and reflective chains, HealthQ assesses how effectively these chains elicit comprehensive and relevant patient information. To achieve this, we integrate an LLM judge to evaluate generated questions across metrics such as specificity, relevance, and usefulness, while aligning these evaluations with traditional Natural Language Processing (NLP) metrics like ROUGE and Named Entity Recognition (NER)-based set comparisons. We validate HealthQ using two custom datasets constructed from public medical datasets, ChatDoctor and MTS-Dialog, and demonstrate its robustness across multiple LLM judge models, including GPT-3.5, GPT-4, and Claude. Our contributions are threefold: we present the first systematic framework for assessing questioning capabilities in healthcare conversations, establish a model-agnostic evaluation methodology, and provide empirical evidence linking high-quality questions to improved patient information elicitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02487v3">LiCoEval: Evaluating LLMs on License Compliance in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 The 47th International Conference on Software Engineering(ICSE 2025)
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have revolutionized code generation, leading to widespread adoption of AI coding tools by developers. However, LLMs can generate license-protected code without providing the necessary license information, leading to potential intellectual property violations during software production. This paper addresses the critical, yet underexplored, issue of license compliance in LLM-generated code by establishing a benchmark to evaluate the ability of LLMs to provide accurate license information for their generated code. To establish this benchmark, we conduct an empirical study to identify a reasonable standard for "striking similarity" that excludes the possibility of independent creation, indicating a copy relationship between the LLM output and certain open-source code. Based on this standard, we propose LiCoEval, to evaluate the license compliance capabilities of LLMs, i.e., the ability to provide accurate license or copyright information when they generate code with striking similarity to already existing copyrighted code. Using LiCoEval, we evaluate 14 popular LLMs, finding that even top-performing LLMs produce a non-negligible proportion (0.88% to 2.01%) of code strikingly similar to existing open-source implementations. Notably, most LLMs fail to provide accurate license information, particularly for code under copyleft licenses. These findings underscore the urgent need to enhance LLM compliance capabilities in code generation tasks. Our study provides a foundation for future research and development to improve license compliance in AI-assisted software development, contributing to both the protection of open-source software copyrights and the mitigation of legal risks for LLM users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17967v1">LLM Knows Geometry Better than Algebra: Numerical Understanding of LLM-Based Agents in A Trading Arena</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly improved performance in natural language processing tasks. However, their ability to generalize to dynamic, unseen tasks, particularly in numerical reasoning, remains a challenge. Existing benchmarks mainly evaluate LLMs on problems with predefined optimal solutions, which may not align with real-world scenarios where clear answers are absent. To bridge this gap, we design the Agent Trading Arena, a virtual numerical game simulating complex economic systems through zero-sum games, where agents invest in stock portfolios. Our experiments reveal that LLMs, including GPT-4o, struggle with algebraic reasoning when dealing with plain-text stock data, often focusing on local details rather than global trends. In contrast, LLMs perform significantly better with geometric reasoning when presented with visual data, such as scatter plots or K-line charts, suggesting that visual representations enhance numerical reasoning. This capability is further improved by incorporating the reflection module, which aids in the analysis and interpretation of complex data. We validate our findings on NASDAQ Stock dataset, where LLMs demonstrate stronger reasoning with visual data compared to text. Our code and data are publicly available at https://github.com/wekjsdvnm/Agent-Trading-Arena.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14804v3">Can LLMs Solve longer Math Word Problems Better?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Math Word Problems (MWPs) play a vital role in assessing the capabilities of Large Language Models (LLMs), yet current research primarily focuses on questions with concise contexts. The impact of longer contexts on mathematical reasoning remains under-explored. This study pioneers the investigation of Context Length Generalizability (CoLeG), which refers to the ability of LLMs to solve MWPs with extended narratives. We introduce Extended Grade-School Math (E-GSM), a collection of MWPs featuring lengthy narratives, and propose two novel metrics to evaluate the efficacy and resilience of LLMs in tackling these problems. Our analysis of existing zero-shot prompting techniques with proprietary LLMs along with open-source LLMs reveals a general deficiency in CoLeG. To alleviate these issues, we propose tailored approaches for different categories of LLMs. For proprietary LLMs, we introduce a new instructional prompt designed to mitigate the impact of long contexts. For open-source LLMs, we develop a novel auxiliary task for fine-tuning to enhance CoLeG. Our comprehensive results demonstrate the effectiveness of our proposed methods, showing improved performance on E-GSM. Additionally, we conduct an in-depth analysis to differentiate the effects of semantic understanding and reasoning efficacy, showing that our methods improves the latter. We also establish the generalizability of our methods across several other MWP benchmarks. Our findings highlight the limitations of current LLMs and offer practical solutions correspondingly, paving the way for further exploration of model generalizability and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17898v1">VeriPlan: Integrating Formal Verification and LLMs into End-User Planning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 In CHI Conference on Human Factors in Computing Systems (CHI '25), April 26-May 1, 2025, Yokohama, Japan. ACM, New York, NY, USA, 19 pages
    </div>
    <details class="paper-abstract">
      Automated planning is traditionally the domain of experts, utilized in fields like manufacturing and healthcare with the aid of expert planning tools. Recent advancements in LLMs have made planning more accessible to everyday users due to their potential to assist users with complex planning tasks. However, LLMs face several application challenges within end-user planning, including consistency, accuracy, and user trust issues. This paper introduces VeriPlan, a system that applies formal verification techniques, specifically model checking, to enhance the reliability and flexibility of LLMs for end-user planning. In addition to the LLM planner, VeriPlan includes three additional core features -- a rule translator, flexibility sliders, and a model checker -- that engage users in the verification process. Through a user study (n=12), we evaluate VeriPlan, demonstrating improvements in the perceived quality, usability, and user satisfaction of LLMs. Our work shows the effective integration of formal verification and user-control features with LLMs for end-user planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17882v1">Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Scientific research is inherently global. However, the vast majority of academic journals are published exclusively in English, creating barriers for non-native-English-speaking researchers. In this study, we leverage large language models (LLMs) to translate published scientific articles while preserving their native JATS XML formatting, thereby developing a practical, automated approach for implementation by academic journals. Using our approach, we translate articles across multiple scientific disciplines into 28 languages. To evaluate translation accuracy, we introduce a novel question-and-answer (QA) benchmarking method, in which an LLM generates comprehension-based questions from the original text and then answers them based on the translated text. Our benchmark results show an average performance of 95.9%, showing that the key scientific details are accurately conveyed. In a user study, we translate the scientific papers of 15 researchers into their native languages, finding that the authors consistently found the translations to accurately capture the original information in their articles. Interestingly, a third of the authors found many technical terms "overtranslated," expressing a preference to keep terminology more familiar in English untranslated. Finally, we demonstrate how in-context learning techniques can be used to align translations with domain-specific preferences such as mitigating overtranslation, highlighting the adaptability and utility of LLM-driven scientific translation. The code and translated articles are available at https://hankleid.github.io/ProjectMundo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17878v1">Towards Enhanced Immersion and Agency for LLM-based Interactive Drama</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      LLM-based Interactive Drama is a novel AI-based dialogue scenario, where the user (i.e. the player) plays the role of a character in the story, has conversations with characters played by LLM agents, and experiences an unfolding story. This paper begins with understanding interactive drama from two aspects: Immersion, the player's feeling of being present in the story, and Agency, the player's ability to influence the story world. Both are crucial to creating an enjoyable interactive experience, while they have been underexplored in previous work. To enhance these two aspects, we first propose Playwriting-guided Generation, a novel method that helps LLMs craft dramatic stories with substantially improved structures and narrative quality. Additionally, we introduce Plot-based Reflection for LLM agents to refine their reactions to align with the player's intentions. Our evaluation relies on human judgment to assess the gains of our methods in terms of immersion and agency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17857v1">SYNTHEMPATHY: A Scalable Empathy Corpus Generated Using LLMs Without Any Crowdsourcing</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Previous research has shown that humans are more receptive towards language models that that exhibit empathetic behavior. While empathy is essential for developing helpful dialogue agents, very few large corpora containing empathetic dialogues are available for fine-tune LLMs. The few existing corpora have largely relied on crowdsourcing to simulate empathetic conversations, a process that is expensive, time-consuming, and not scalable to larger datasets. We propose a data generation framework for developing SYNTHEMPATHY, a large corpus containing 105k empathetic responses to real-life situations compiled through LLM generation. A base Mistral 7B model fine-tuned on our SYNTHEMPATHY corpus exhibits an increase in the average empathy score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06364v2">Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Adapting pre-trained large language models (LLMs) is crucial but challenging due to their enormous size. Parameter-efficient fine-tuning (PEFT) techniques typically employ additive adapters applied to frozen model weights. To further reduce memory usage, model weights can be compressed through quantization. However, existing PEFT methods often yield suboptimal model quality due to restrictive assumptions, such as imposing low-rank constraints on adapters to reduce trainable parameters. We find that sketching, a popular data compression technique, can serve as an efficient adaptation strategy for LLMs while avoiding low-rank assumptions. We introduce SketchTune, a compressive adaptation strategy that compresses LLM weights into compact fine-tunable sketches, integrating compression and adaptation into a unified framework. This integration eliminates the need for complex two-path computation common in existing PEFT techniques, enabling faster and more memory-efficient training and inference. SketchTune is supported by mathematical insights into matrix classes that are better approximated using sketching rather than low-rank methods. Our rigorous evaluations with Llama-1/2/3 models demonstrate that SketchTune outperforms leading PEFT methods across diverse tasks including math problem-solving, common sense reasoning, and instruction following, while using substantially smaller base models and comparable trainable parameters. As a highlight, SketchTune outperforms LoRA, DoRA, and S2FT on commonsense and math benchmarks using 2.6-3.5$\times$ smaller base models and exceeds LoftQ in accuracy by 14.48% on GSM8K with 7.3$\times$ fewer trainable parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17823v1">A General Framework to Enhance Fine-tuning-based LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Unlearning has been proposed to remove copyrighted and privacy-sensitive data from Large Language Models (LLMs). Existing approaches primarily rely on fine-tuning-based methods, which can be categorized into gradient ascent-based (GA-based) and suppression-based methods. However, they often degrade model utility (the ability to respond to normal prompts). In this work, we aim to develop a general framework that enhances the utility of fine-tuning-based unlearning methods. To achieve this goal, we first investigate the common property between GA-based and suppression-based methods. We unveil that GA-based methods unlearn by distinguishing the target data (i.e., the data to be removed) and suppressing related generations, which is essentially the same strategy employed by suppression-based methods. Inspired by this finding, we introduce Gated Representation UNlearning (GRUN) which has two components: a soft gate function for distinguishing target data and a suppression module using Representation Fine-tuning (ReFT) to adjust representations rather than model parameters. Experiments show that GRUN significantly improves the unlearning and utility. Meanwhile, it is general for fine-tuning-based methods, efficient and promising for sequential unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09179v2">Towards Effective Evaluations and Comparisons for LLM Unlearning Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The imperative to eliminate undesirable data memorization underscores the significance of machine unlearning for large language models (LLMs). Recent research has introduced a series of promising unlearning methods, notably boosting the practical significance of the field. Nevertheless, adopting a proper evaluation framework to reflect the true unlearning efficacy is also essential yet has not received adequate attention. This paper seeks to refine the evaluation of LLM unlearning by addressing two key challenges -- a) the robustness of evaluation metrics and b) the trade-offs between competing goals. The first challenge stems from findings that current metrics are susceptible to various red teaming scenarios. It indicates that they may not reflect the true extent of knowledge retained by LLMs but rather tend to mirror superficial model behaviors, thus prone to attacks. We address this issue by devising and assessing a series of candidate metrics, selecting the most robust ones under various types of attacks. The second challenge arises from the conflicting goals of eliminating unwanted knowledge while retaining those of others. This trade-off between unlearning and retention often fails to conform the Pareto frontier, rendering it subtle to compare the efficacy between methods that excel only in either unlearning or retention. We handle this issue by proposing a calibration method that can restore the original performance on non-targeted data after unlearning, thereby allowing us to focus exclusively on assessing the strength of unlearning. Our evaluation framework notably enhances the effectiveness when assessing and comparing various LLM unlearning methods, further allowing us to benchmark existing works, identify their proper hyper-parameters, and explore new tricks to enhance their practical efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04420v3">KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 36 pages. Code: https://github.com/cmd2001/KVTuner
    </div>
    <details class="paper-abstract">
      KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we thoroughly analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 38.3% compared with KV8 quantization over various context lengths. Our code and searched configurations are available at https://github.com/cmd2001/KVTuner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17812v1">Can Multimodal LLMs Perform Time Series Anomaly Detection?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 9 pages for the main content; 32 pages for the full paper including the appendix. More resources on the intersection of multimodal LLMs and time series analysis are on the website https://mllm-ts.github.io
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been increasingly used in time series analysis. However, the potential of multimodal LLMs (MLLMs), particularly vision-language models, for time series remains largely under-explored. One natural way for humans to detect time series anomalies is through visualization and textual description. Motivated by this, we raise a critical and practical research question: Can multimodal LLMs perform time series anomaly detection? To answer this, we propose VisualTimeAnomaly benchmark to evaluate MLLMs in time series anomaly detection (TSAD). Our approach transforms time series numerical data into the image format and feed these images into various MLLMs, including proprietary models (GPT-4o and Gemini-1.5) and open-source models (LLaVA-NeXT and Qwen2-VL), each with one larger and one smaller variant. In total, VisualTimeAnomaly contains 12.4k time series images spanning 3 scenarios and 3 anomaly granularities with 9 anomaly types across 8 MLLMs. Starting with the univariate case (point- and range-wise anomalies), we extend our evaluation to more practical scenarios, including multivariate and irregular time series scenarios, and variate-wise anomalies. Our study reveals several key insights: 1) MLLMs detect range- and variate-wise anomalies more effectively than point-wise anomalies. 2) MLLMs are highly robust to irregular time series, even with 25% of the data missing. 3) Open-source MLLMs perform comparably to proprietary models in TSAD. While open-source MLLMs excel on univariate time series, proprietary MLLMs demonstrate superior effectiveness on multivariate time series. To the best of our knowledge, this is the first work to comprehensively investigate MLLMs for TSAD, particularly for multivariate and irregular time series scenarios. We release our dataset and code at https://github.com/mllm-ts/VisualTimeAnomaly to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10573v2">Putting People in LLMs' Shoes: Generating Better Answers via Question Rewriter</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 7 pages, 4 figures, 5 tables and accepted at AAAI 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant capabilities, particularly in the domain of question answering (QA). However, their effectiveness in QA is often undermined by the vagueness of user questions. To address this issue, we introduce single-round instance-level prompt optimization, referred to as question rewriter. By enhancing the intelligibility of human questions for black-box LLMs, our question rewriter improves the quality of generated answers. The rewriter is optimized using direct preference optimization based on feedback collected from automatic criteria for evaluating generated answers; therefore, its training does not require costly human annotations. The experiments across multiple black-box LLMs and long-form question answering (LFQA) datasets demonstrate the efficacy of our method. This paper provides a practical framework for training question rewriters and sets a precedent for future explorations in prompt optimization within LFQA tasks. Code is available at https://github.com/3244we/Question-Rewriter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17773v1">Uncertainty Quantification for LLM-Based Survey Simulations</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 30 pages, 6 figures, 10 tables
    </div>
    <details class="paper-abstract">
      We investigate the reliable use of simulated survey responses from large language models (LLMs) through the lens of uncertainty quantification. Our approach converts synthetic data into confidence sets for population parameters of human responses, addressing the distribution shift between the simulated and real populations. A key innovation lies in determining the optimal number of simulated responses: too many produce overly narrow confidence sets with poor coverage, while too few yield excessively loose estimates. To resolve this, our method adaptively selects the simulation sample size, ensuring valid average-case coverage guarantees. It is broadly applicable to any LLM, irrespective of its fidelity, and any procedure for constructing confidence sets. Additionally, the selected sample size quantifies the degree of misalignment between the LLM and the target human population. We illustrate our method on real datasets and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13704v2">DHP Benchmark: Are LLMs Good NLG Evaluators?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly serving as evaluators in Natural Language Generation (NLG) tasks; this is often referred to as ``LLM-as-a-judge'' paradigm. However, the capabilities of LLMs in evaluating NLG quality remain underexplored. Current studies depend on human assessments and simple metrics that fail to capture the discernment of LLMs across diverse NLG tasks. To address this gap, we propose the Discernment of Hierarchical Perturbation (DHP) benchmarking framework, which provides quantitative discernment scores for LLMs. This framework leverages hierarchically perturbed text data and statistical tests to systematically measure the NLG evaluation capabilities of LLMs. We re-established six evaluation datasets for this benchmark, covering four NLG tasks: Summarization, Story Completion, Question Answering, and Translation. Our comprehensive benchmarking of five major LLM families provides critical insight into their strengths and limitations as NLG evaluators. Our dataset is available at https://huggingface.co/datasets/YCWANGVINCE/DHP_Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17763v1">Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Traditional security protection methods struggle to address sophisticated attack vectors in large-scale distributed systems, particularly when balancing detection accuracy with data privacy concerns. This paper presents a novel distributed security threat detection system that integrates federated learning with multimodal large language models (LLMs). Our system leverages federated learning to ensure data privacy while employing multimodal LLMs to process heterogeneous data sources including network traffic, system logs, images, and sensor data. Experimental evaluation on a 10TB distributed dataset demonstrates that our approach achieves 96.4% detection accuracy, outperforming traditional baseline models by 4.1 percentage points. The system reduces both false positive and false negative rates by 1.8 and 2.4 percentage points respectively. Performance analysis shows that our system maintains efficient processing capabilities in distributed environments, requiring 180 seconds for model training and 3.8 seconds for threat detection across the distributed network. These results demonstrate significant improvements in detection accuracy and computational efficiency while preserving data privacy, suggesting strong potential for real-world deployment in large-scale security systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17749v1">Detection of LLM-Paraphrased Code and Identification of the Responsible LLM Using Coding Style Features</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) for code generation has raised serious concerns about intellectual property protection. Malicious users can exploit LLMs to produce paraphrased versions of proprietary code that closely resemble the original. While the potential for LLM-assisted code paraphrasing continues to grow, research on detecting it remains limited, underscoring an urgent need for detection system. We respond to this need by proposing two tasks. The first task is to detect whether code generated by an LLM is a paraphrased version of original human-written code. The second task is to identify which LLM is used to paraphrase the original code. For these tasks, we construct a dataset LPcode consisting of pairs of human-written code and LLM-paraphrased code using various LLMs. We statistically confirm significant differences in the coding styles of human-written and LLM-paraphrased code, particularly in terms of naming consistency, code structure, and readability. Based on these findings, we develop LPcodedec, a detection method that identifies paraphrase relationships between human-written and LLM-generated code, and discover which LLM is used for the paraphrasing. LPcodedec outperforms the best baselines in two tasks, improving F1 scores by 2.64% and 15.17% while achieving speedups of 1,343x and 213x, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17428v3">NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 ICLR 2025 (Spotlight). We open-source the model at: https://huggingface.co/nvidia/NV-Embed-v2
    </div>
    <details class="paper-abstract">
      Decoder-only LLM-based embedding models are beginning to outperform BERT or T5-based embedding models in general-purpose text embedding tasks, including dense vector-based retrieval. In this work, we introduce NV-Embed, incorporating architectural designs, training procedures, and curated datasets to significantly enhance the performance of LLM as a versatile embedding model, while maintaining its simplicity and reproducibility. For model architecture, we propose a latent attention layer to obtain pooled embeddings, which consistently improves retrieval and downstream task accuracy compared to mean pooling or using the last <EOS> token embedding from LLMs. To enhance representation learning, we remove the causal attention mask of LLMs during contrastive training. For training algorithm, we introduce a two-stage contrastive instruction-tuning method. It first applies contrastive training with instructions on retrieval datasets, utilizing in-batch negatives and curated hard negative examples. At stage-2, it blends various non-retrieval into instruction tuning, which not only enhances non-retrieval task accuracy but also improves retrieval performance. For training data, we utilize the hard-negative mining, synthetic data generation and existing public available datasets to boost the performance of embedding model. By combining these techniques, our NV-Embed-v1 and NV-Embed-v2 models obtained the No.1 position on the MTEB leaderboard (as of May 24 and August 30, 2024, respectively) across 56 tasks, demonstrating the sustained effectiveness of the proposed methods over time. It also achieved the highest scores in the Long Doc section and the second-highest scores in the QA section of the AIR Benchmark, which covers a range of out-of-domain information retrieval topics beyond those in MTEB. We further provide the analysis of model compression techniques for generalist embedding models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v2">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding: it asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15097v2">LUME: LLM Unlearning with Multitask Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Unlearning aims to remove copyrighted, sensitive, or private content from large language models (LLMs) without a full retraining. In this work, we develop a multi-task unlearning benchmark (LUME) which features three tasks: (1) unlearn synthetically generated creative short novels, (2) unlearn synthetic biographies with sensitive information, and (3) unlearn a collection of public biographies. We further release two fine-tuned LLMs of 1B and 7B parameter sizes as the target models. We conduct detailed evaluations of several recently proposed unlearning algorithms and present results on carefully crafted metrics to understand their behavior and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18650v1">Single- vs. Dual-Prompt Dialogue Generation with LLMs for Job Interviews in Human Resources</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Optimizing language models for use in conversational agents requires large quantities of example dialogues. Increasingly, these dialogues are synthetically generated by using powerful large language models (LLMs), especially in domains with challenges to obtain authentic human data. One such domain is human resources (HR). In this context, we compare two LLM-based dialogue generation methods for the use case of generating HR job interviews, and assess whether one method generates higher-quality dialogues that are more challenging to distinguish from genuine human discourse. The first method uses a single prompt to generate the complete interview dialog. The second method uses two agents that converse with each other. To evaluate dialogue quality under each method, we ask a judge LLM to determine whether AI was used for interview generation, using pairwise interview comparisons. We demonstrate that despite a sixfold increase in token cost, interviews generated with the dual-prompt method achieve a win rate up to ten times higher than those generated with the single-prompt method. This difference remains consistent regardless of whether GPT-4o or Llama 3.3 70B is used for either interview generation or judging quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18635v1">Faster, Cheaper, Better: Multi-Objective Hyperparameter Optimization for LLM and RAG Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      While Retrieval Augmented Generation (RAG) has emerged as a popular technique for improving Large Language Model (LLM) systems, it introduces a large number of choices, parameters and hyperparameters that must be made or tuned. This includes the LLM, embedding, and ranker models themselves, as well as hyperparameters governing individual RAG components. Yet, collectively optimizing the entire configuration in a RAG or LLM system remains under-explored - especially in multi-objective settings - due to intractably large solution spaces, noisy objective evaluations, and the high cost of evaluations. In this work, we introduce the first approach for multi-objective parameter optimization of cost, latency, safety and alignment over entire LLM and RAG systems. We find that Bayesian optimization methods significantly outperform baseline approaches, obtaining a superior Pareto front on two new RAG benchmark tasks. We conclude our work with important considerations for practitioners who are designing multi-objective RAG systems, highlighting nuances such as how optimal configurations may not generalize across tasks and objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18532v1">CuDIP: Enhancing Theorem Proving in LLMs via Curriculum Learning-based Direct Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Automated theorem proving (ATP) is one of the most challenging mathematical reasoning tasks for Large Language Models (LLMs). Most existing LLM-based ATP methods rely on supervised fine-tuning, which results in a limited alignment between the theorem proving process and human preferences. Direct Preference Optimization (DPO), which aligns LLMs with human preferences, has shown positive effects for certain tasks. However, the lack of high-quality preference data for theorem proving presents a significant challenge. In this paper, we innovatively apply DPO to formal automated theorem proving and introduces a Curriculum Learning-based DPO Iterative Theorem Proving (CuDIP) method. Specifically, we propose a method for constructing preference data which utilizes LLMs and existing theorem proving data to enhance the diversity of the preference data while reducing the reliance on human preference annotations. We then integrate this preference data construction method with curriculum learning to iteratively fine-tune the theorem proving model through DPO. Experimental results on the MiniF2F and ProofNet datasets demonstrate the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18530v1">IMPROVE: Iterative Model Pipeline Refinement and Optimization Leveraging LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Computer vision is a critical component in a wide range of real-world applications, including plant monitoring in agriculture and handwriting classification in digital systems. However, developing high-performance computer vision models traditionally demands both machine learning (ML) expertise and domain-specific knowledge, making the process costly, labor-intensive, and inaccessible to many. Large language model (LLM) agents have emerged as a promising solution to automate this workflow, but most existing methods share a common limitation: they attempt to optimize entire pipelines in a single step before evaluation, making it difficult to attribute improvements to specific changes. This lack of granularity leads to unstable optimization and slower convergence, limiting their effectiveness. To address this, we introduce Iterative Refinement, a novel strategy for LLM-driven ML pipeline design inspired by how human ML experts iteratively refine models, focusing on one component at a time rather than making sweeping changes all at once. By systematically updating individual components based on real training feedback, Iterative Refinement improves stability, interpretability, and overall model performance. We implement this strategy in IMPROVE, an end-to-end LLM agent framework for automating and optimizing object classification pipelines. Through extensive evaluations across datasets of varying sizes and domains, including standard benchmarks and Kaggle competition datasets, we demonstrate that Iterative Refinement enables IMPROVE to consistently achieve better performance over existing zero-shot LLM-based approaches. These findings establish Iterative Refinement as an effective new strategy for LLM-driven ML automation and position IMPROVE as an accessible solution for building high-quality computer vision models without requiring ML expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v1">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding: it asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17410v1">COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 23 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across various domains, yet their optimization remains a significant challenge due to the complex and high-dimensional loss landscapes they inhabit. While adaptive optimizers such as AdamW are widely used, they suffer from critical limitations, including an inability to capture interdependencies between coordinates and high memory consumption. Subsequent research, exemplified by SOAP, attempts to better capture coordinate interdependence but incurs greater memory overhead, limiting scalability for massive LLMs. An alternative approach aims to reduce memory consumption through low-dimensional projection, but this leads to substantial approximation errors, resulting in less effective optimization (e.g., in terms of per-token efficiency). In this paper, we propose COSMOS, a novel hybrid optimizer that leverages the varying importance of eigensubspaces in the gradient matrix to achieve memory efficiency without compromising optimization performance. The design of COSMOS is motivated by our empirical insights and practical considerations. Specifically, COSMOS applies SOAP to the leading eigensubspace, which captures the primary optimization dynamics, and MUON to the remaining eigensubspace, which is less critical but computationally expensive to handle with SOAP. This hybrid strategy significantly reduces memory consumption while maintaining robust optimization performance, making it particularly suitable for massive LLMs. Numerical experiments on various datasets and transformer architectures are provided to demonstrate the effectiveness of COSMOS. Our code is available at https://github.com/lliu606/COSMOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19381v4">HYBRIDMIND: Meta Selection of Natural Language and Symbolic Language for Enhanced LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      LLMs approach logical and mathematical reasoning through natural or symbolic languages. While natural language offers human-accessible flexibility but suffers from ambiguity, symbolic reasoning provides precise, machine-executable inferences at the cost of strict domain constraints. We introduce HYBRIDMIND, an adaptive strategy that selects the optimal reasoning approach for each reasoning problem. Through extensive experiments, we evaluate both prompting-based approaches with state-of-the-art LLMs and fine-tuned open-source models. We find that fine-tuning LLaMA-3.1-8B-Instruct as a meta-selector outperforms GPT-4o's natural language reasoning by 4.4\% on FOLIO and 1.3\% on MATH. More notably, using GPT-3.5-turbo as a prompted meta-selector yields a 10\% improvement on FOLIO's challenging subset compared to GPT-4o. We will release our code and data to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05315v3">Aligned at the Start: Conceptual Groupings in LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      This paper shifts focus to the often-overlooked input embeddings - the initial representations fed into transformer blocks. Using fuzzy graph, k-nearest neighbor (k-NN), and community detection, we analyze embeddings from diverse LLMs, finding significant categorical community structure aligned with predefined concepts and categories aligned with humans. We observe these groupings exhibit within-cluster organization (such as hierarchies, topological ordering, etc.), hypothesizing a fundamental structure that precedes contextual processing. To further investigate the conceptual nature of these groupings, we explore cross-model alignments across different LLM categories within their input embeddings, observing a medium to high degree of alignment. Furthermore, provide evidence that manipulating these groupings can play a functional role in mitigating ethnicity bias in LLM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06287v2">Non-Halting Queries: Exploiting Fixed Points in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      We introduce a new vulnerability that exploits fixed points in autoregressive models and use it to craft queries that never halt. More precisely, for non-halting queries, the LLM never samples the end-of-string token <eos>. We rigorously analyze the conditions under which the non-halting anomaly presents itself. In particular, at temperature zero, we prove that if a repeating (cyclic) token sequence is observed at the output beyond the context size, then the LLM does not halt. We demonstrate non-halting queries in many experiments performed in base unaligned models where repeating prompts immediately lead to a non-halting cyclic behavior as predicted by the analysis. Further, we develop a simple recipe that takes the same fixed points observed in the base model and creates a prompt structure to target aligned models. We demonstrate the recipe's success in sending every major model released over the past year into a non-halting state with the same simple prompt even over higher temperatures. Further, we devise an experiment with 100 randomly selected tokens and show that the recipe to create non-halting queries succeeds with high success rates ranging from 97% for GPT-4o to 19% for Gemini Pro 1.5. These results show that the proposed adversarial recipe succeeds in bypassing alignment at one to two orders of magnitude higher rates compared to earlier reports. We also study gradient-based direct inversion using ARCA to craft new short prompts to induce the non-halting state. We inverted 10,000 random repeating 2-cycle outputs for llama-3.1-8b-instruct. Out of 10,000 three-token inverted prompts 1,512 yield non-halting queries reaching a rate of 15%. Our experiments with ARCA show that non-halting may be easily induced with as few as 3 input tokens with high probability. Overall, our experiments demonstrate that non-halting queries are prevalent and relatively easy to find.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17341v1">Time series forecasting based on optimized LLM for fault prediction in distribution power grid insulators</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Surface contamination on electrical grid insulators leads to an increase in leakage current until an electrical discharge occurs, which can result in a power system shutdown. To mitigate the possibility of disruptive faults resulting in a power outage, monitoring contamination and leakage current can help predict the progression of faults. Given this need, this paper proposes a hybrid deep learning (DL) model for predicting the increase in leakage current in high-voltage insulators. The hybrid structure considers a multi-criteria optimization using tree-structured Parzen estimation, an input stage filter for signal noise attenuation combined with a large language model (LLM) applied for time series forecasting. The proposed optimized LLM outperforms state-of-the-art DL models with a root-mean-square error equal to 2.24$\times10^{-4}$ for a short-term horizon and 1.21$\times10^{-3}$ for a medium-term horizon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17328v1">Mutual Reinforcement of LLM Dialogue Synthesis and Summarization Capabilities for Few-Shot Dialogue Summarization</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      In this work, we propose Mutual Reinforcing Data Synthesis (MRDS) within LLMs to improve few-shot dialogue summarization task. Unlike prior methods that require external knowledge, we mutually reinforce the LLM\'s dialogue synthesis and summarization capabilities, allowing them to complement each other during training and enhance overall performances. The dialogue synthesis capability is enhanced by directed preference optimization with preference scoring from summarization capability. The summarization capability is enhanced by the additional high quality dialogue-summary paired data produced by the dialogue synthesis capability. By leveraging the proposed MRDS mechanism, we elicit the internal knowledge of LLM in the format of synthetic data, and use it to augment the few-shot real training dataset. Empirical results demonstrate that our method improves dialogue summarization, achieving a 1.5% increase in ROUGE scores and a 0.3% improvement in BERT scores in few-shot settings. Furthermore, our method attains the highest average scores in human evaluations, surpassing both the pre-trained models and the baselines fine-tuned solely for summarization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17304v1">Child vs. machine language learning: Can the logical structure of human language unleash LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 ISCA/ITG Workshop on Diversity in Large Speech and Language Models
    </div>
    <details class="paper-abstract">
      We argue that human language learning proceeds in a manner that is different in nature from current approaches to training LLMs, predicting a difference in learning biases. We then present evidence from German plural formation by LLMs that confirm our hypothesis that even very powerful implementations produce results that miss aspects of the logic inherent to language that humans have no problem with. We conclude that attention to the different structures of human language and artificial neural networks is likely to be an avenue to improve LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04420v2">KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 36 pages. Code: https://github.com/cmd2001/KVTuner
    </div>
    <details class="paper-abstract">
      KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we thoroughly analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 38.3% compared with KV8 quantization over various context lengths. Our code and searched configurations are available at https://github.com/cmd2001/KVTuner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17298v1">Delta Decompression for MoE-based LLMs Compression</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures in large language models (LLMs) achieve exceptional performance, but face prohibitive storage and memory requirements. To address these challenges, we present $D^2$-MoE, a new delta decompression compressor for reducing the parameters of MoE LLMs. Based on observations of expert diversity, we decompose their weights into a shared base weight and unique delta weights. Specifically, our method first merges each expert's weight into the base weight using the Fisher information matrix to capture shared components. Then, we compress delta weights through Singular Value Decomposition (SVD) by exploiting their low-rank properties. Finally, we introduce a semi-dynamical structured pruning strategy for the base weights, combining static and dynamic redundancy analysis to achieve further parameter reduction while maintaining input adaptivity. In this way, our $D^2$-MoE successfully compact MoE LLMs to high compression ratios without additional training. Extensive experiments highlight the superiority of our approach, with over 13% performance gains than other compressors on Mixtral|Phi-3.5|DeepSeek|Qwen2 MoE LLMs at 40$\sim$60% compression rates. Codes are available in https://github.com/lliai/D2MoE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17282v1">Capability Instruction Tuning: A New Paradigm for Dynamic LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 AAAI 2025; Project Page: https://cit-llm-routing.github.io
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated human-like instruction-following abilities, particularly those exceeding 100 billion parameters. The combined capability of some smaller, resource-friendly LLMs can address most of the instructions that larger LLMs excel at. In this work, we explore how to route the best-performing LLM for each instruction to achieve better overall performance. We develop a new paradigm, constructing capability instructions with model capability representation, user instruction, and performance inquiry prompts to assess the performance. To learn from capability instructions, we introduce a new end-to-end framework called Model Selection with Aptitude Test (Model-SAT), which generates positive and negative samples based on what different models perform well or struggle with. Model-SAT uses a model capability encoder that extends its model representation to a lightweight LLM. Our experiments show that Model-SAT understands the performance dimensions of candidate models and provides the probabilities of their capability to handle various instructions. Additionally, during deployment, a new model can quickly infer its aptitude test results across 50 tasks, each with 20 shots. Model-SAT performs state-of-the-art model routing without candidate inference and in real-world new model-released scenarios. The code is available at https://github.com/Now-Join-Us/CIT-LLM-Routing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20296v2">PersonalLLM: Tailoring LLMs to Individual Preferences</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 28 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As LLMs become capable of complex tasks, there is growing potential for personalized interactions tailored to the subtle and idiosyncratic preferences of the user. We present a public benchmark, PersonalLLM, focusing on adapting LLMs to provide maximal benefits for a particular user. Departing from existing alignment benchmarks that implicitly assume uniform preferences, we curate open-ended prompts paired with many high-quality answers over which users would be expected to display heterogeneous latent preferences. Instead of persona-prompting LLMs based on high-level attributes (e.g., user's race or response length), which yields homogeneous preferences relative to humans, we develop a method that can simulate a large user base with diverse preferences from a set of pre-trained reward models. Our dataset and generated personalities offer an innovative testbed for developing personalization algorithms that grapple with continual data sparsity--few relevant feedback from the particular user--by leveraging historical data from other (similar) users. We explore basic in-context learning and meta-learning baselines to illustrate the utility of PersonalLLM and highlight the need for future methodological development. Our dataset is available at https://huggingface.co/datasets/namkoong-lab/PersonalLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17262v1">Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 21 pages,6 figures
    </div>
    <details class="paper-abstract">
      The rapid advancements in computing dramatically increase the scale and cost of training Large Language Models (LLMs). Accurately predicting downstream task performance prior to model training is crucial for efficient resource allocation, yet remains challenging due to two primary constraints: (1) the "emergence phenomenon", wherein downstream performance metrics become meaningful only after extensive training, which limits the ability to use smaller models for prediction; (2) Uneven task difficulty distributions and the absence of consistent scaling laws, resulting in substantial metric variability. Existing performance prediction methods suffer from limited accuracy and reliability, thereby impeding the assessment of potential LLM capabilities. To address these challenges, we propose a Clustering-On-Difficulty (COD) downstream performance prediction framework. COD first constructs a predictable support subset by clustering tasks based on difficulty features, strategically excluding non-emergent and non-scalable clusters. The scores on the selected subset serve as effective intermediate predictors of downstream performance on the full evaluation set. With theoretical support, we derive a mapping function that transforms performance metrics from the predictable subset to the full evaluation set, thereby ensuring accurate extrapolation of LLM downstream performance. The proposed method has been applied to predict performance scaling for a 70B LLM, providing actionable insights for training resource allocation and assisting in monitoring the training process. Notably, COD achieves remarkable predictive accuracy on the 70B LLM by leveraging an ensemble of small models, demonstrating an absolute mean deviation of 1.36% across eight important LLM evaluation benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17216v1">Making LLMs Reason? The Intermediate Language Problem in Neurosymbolic Approaches</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Logical reasoning tasks manifest themselves as a challenge to Large Language Models (LLMs). Neurosymbolic approaches use LLMs to translate logical reasoning problems formulated in natural language into a formal intermediate language. Subsequently, the usage of symbolic reasoners yields reliable solving thereof. However, LLMs often fail in translation due to poorly chosen intermediate languages. We introduce the intermediate language problem, which is the problem of choosing a suitable formal language representation for neurosymbolic approaches. Theoretically, we argue that its origins lie in the inability of LLMs to distinguish syntax from semantics and the relative independence of the problem from its representation. We showcase its existence experimentally by contrasting two intermediate languages, Answer Set Programming and the Python Knowledge Engine. In addition, we demonstrate the effects of varying degrees of supplementary context information. Our results show a maximum difference in overall-accuracy of 53.20% and 49.26% in execution-accuracy. When using the GPT4o-mini LLM we beat the state-of-the-art in overall-accuracy on the ProntoQA dataset by 21.20% and by 50.50% on the ProofWriter dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17214v1">CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in many tasks but struggle to accurately quantify uncertainty in their generated responses. This limitation makes it challenging to detect misinformation and ensure reliable decision-making. Existing uncertainty quantification (UQ) methods for LLMs are primarily prompt-wise rather than response-wise, often requiring multiple response samples, which incurs high computational costs. Moreover, LLMs have been shown to be overconfident, particularly when using reasoning steps to derive their answers. In this work, we propose CoT-UQ, a response-wise UQ framework that integrates LLMs' inherent reasoning capabilities through Chain-of-Thought (CoT) into the UQ process. CoT-UQ captures critical information during inference by extracting keywords from each reasoning step and assessing their importance to the final answer. This key reasoning information is then aggregated to produce a final uncertainty estimate. We conduct extensive experiments based on LLaMA Family with model sizes varying from 8B to 13B across logical and mathematical reasoning tasks. Experimental results demonstrate that CoT-UQ significantly outperforms existing UQ methods, achieving an average improvement of 5.9% AUROC compared to current UQ methods. The code is available at: https://github.com/ZBox1005/CoT-UQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17187v1">Evaluating Expert Contributions in a MoE LLM for Quiz-Based Tasks</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 preprint, short paper
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) with Mixture of Experts (MoE) layers have gained significant attention. Currently, state-of-the-art LLMs utilize this architecture. There is a substantial amount of research on how to train such models and how to select hyperparameters for this architecture. However, there is a lack of studies focusing on post-evaluation analysis of MoE layer properties. In this paper, we take a first step toward closing this gap by evaluating expert contributions on the quiz-based MMLU benchmark. We show that most experts were never activated during inference on this benchmark. Additionally, the output distribution of gating networks is much closer to uniform than sparse. Finally, we demonstrate that the average performance of some experts within the same layer varies significantly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08090v2">Multilingual LLMs Inherently Reward In-Language Time-Sensitive Semantic Alignment for Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      The unwavering disparity in labeled resources between resource-rich languages and those considered low-resource remains a significant impediment for Large Language Models (LLMs). Recent strides in cross-lingual in-context learning (X-ICL), mainly through semantically aligned examples retrieved from multilingual pre-trained transformers, have shown promise in mitigating this issue. However, our investigation reveals that LLMs intrinsically reward in-language semantically aligned cross-lingual instances over direct cross-lingual semantic alignments, with a pronounced disparity in handling time-sensitive queries in the X-ICL setup. Such queries demand sound temporal reasoning ability from LLMs, yet the advancements have predominantly focused on English. This study aims to bridge this gap by improving temporal reasoning capabilities in low-resource languages. To this end, we introduce mTEMPREASON, a temporal reasoning dataset aimed at the varied degrees of low-resource languages and propose Cross-Lingual Time-Sensitive Semantic Alignment (CLiTSSA), a novel method to improve temporal reasoning in these contexts. To facilitate this, we construct an extension of mTEMPREASON comprising pairs of parallel cross-language temporal queries along with their anticipated in-language semantic similarity scores. Our empirical evidence underscores the superior performance of CLiTSSA compared to established baselines across three languages -- Romanian, German, and French, encompassing three temporal tasks and including a diverse set of four contemporaneous LLMs. This marks a significant step forward in addressing resource disparity in the context of temporal reasoning across languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18596v2">DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      We introduce DeltaLLM, a new post-training compression technique to reduce the memory footprint of LLMs. We propose an alternative way of structuring LLMs with weight sharing between layers in subsequent Transformer blocks, along with additional low-rank difference matrices between them. For training, we adopt the progressing module replacement method and show that the lightweight training of the low-rank modules with approximately 30M-40M tokens is sufficient to achieve performance on par with LLMs of comparable sizes trained from scratch. We release the resultant models, DeltaLLAMA and DeltaPHI, with a 12% parameter reduction, retaining 90% of the performance of the base Llama and Phi models on common knowledge and reasoning benchmarks. Our method also outperforms compression techniques JointDrop, LaCo, ShortGPT and SliceGPT with the same number of parameters removed. For example, DeltaPhi 2.9B with a 24% reduction achieves similar average zero-shot accuracies as recovery fine-tuned SlicedPhi 3.3B with a 12% reduction, despite being approximately 400M parameters smaller with no fine-tuning applied. This work provides new insights into LLM architecture design and compression methods when storage space is critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15294v2">Round Attention: A Novel Round-Level Attention Mechanism to Accelerate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      The increasing context window size in large language models (LLMs) has improved their ability to handle complex, long-text tasks. However, as the conversation rounds continue, it is required to store a large amount of KV cache in GPU memory, which significantly affects the efficiency and even availability of the model serving systems. This paper analyzes dialogue data from real users and discovers that the LLM inference manifests a watershed layer, after which the distribution of round-level attention shows notable similarity. We propose Round Attention, a novel round-level attention mechanism that only recalls and computes the KV cache of the most relevant rounds. The experiments show that our method saves 55\% memory usage without compromising model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17139v1">CodeSwift: Accelerating LLM Inference for Efficient Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Code generation is a latency-sensitive task that demands high timeliness, but the autoregressive decoding mechanism of Large Language Models (LLMs) leads to poor inference efficiency. Existing LLM inference acceleration methods mainly focus on standalone functions using only built-in components. Moreover, they treat code like natural language sequences, ignoring its unique syntax and semantic characteristics. As a result, the effectiveness of these approaches in code generation tasks remains limited and fails to align with real-world programming scenarios. To alleviate this issue, we propose CodeSwift, a simple yet highly efficient inference acceleration approach specifically designed for code generation, without comprising the quality of the output. CodeSwift constructs a multi-source datastore, providing access to both general and project-specific knowledge, facilitating the retrieval of high-quality draft sequences. Moreover, CodeSwift reduces retrieval cost by controlling retrieval timing, and enhances efficiency through parallel retrieval and a context- and LLM preference-aware cache. Experimental results show that CodeSwift can reach up to 2.53x and 2.54x speedup compared to autoregressive decoding in repository-level and standalone code generation tasks, respectively, outperforming state-of-the-art inference acceleration approaches by up to 88%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17091v1">WildFrame: Comparing Framing in Humans and LLMs on Naturally Occurring Texts</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Humans are influenced by how information is presented, a phenomenon known as the framing effect. Previous work has shown that LLMs may also be susceptible to framing but has done so on synthetic data and did not compare to human behavior. We introduce WildFrame, a dataset for evaluating LLM responses to positive and negative framing, in naturally-occurring sentences, and compare humans on the same data. WildFrame consists of 1,000 texts, first selecting real-world statements with clear sentiment, then reframing them in either positive or negative light, and lastly, collecting human sentiment annotations. By evaluating eight state-of-the-art LLMs on WildFrame, we find that all models exhibit framing effects similar to humans ($r\geq0.57$), with both humans and models being more influenced by positive rather than negative reframing. Our findings benefit model developers, who can either harness framing or mitigate its effects, depending on the downstream application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05806v3">CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Ongoing work; project website is available at https://zjunlp.github.io/project/CKnowEdit code and dataset are available at https://github.com/zjunlp/EasyEdit
    </div>
    <details class="paper-abstract">
      Chinese, as a linguistic system rich in depth and complexity, is characterized by distinctive elements such as ancient poetry, proverbs, idioms, and other cultural constructs. However, current Large Language Models (LLMs) face limitations in these specialized domains, highlighting the need for the development of comprehensive datasets that can assess, continuously update, and progressively improve these culturally-grounded linguistic competencies through targeted training optimizations. To address this gap, we introduce CKnowEdit, the first-ever Chinese knowledge editing dataset designed to correct linguistic, factual, and logical errors in LLMs. We collect seven types of knowledge from a wide range of sources, including classical texts, idioms, and content from Baidu Tieba Ruozhiba, taking into account the unique polyphony, antithesis, and logical structures inherent in the Chinese language. By analyzing this dataset, we highlight the challenges current LLMs face in mastering Chinese. Furthermore, our evaluation of state-of-the-art knowledge editing techniques reveals opportunities to advance the correction of Chinese knowledge. Code and dataset are available at https://github.com/zjunlp/EasyEdit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03438v2">BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have spurred growing interest in automatic theorem proving using Lean4, where effective tree search methods are crucial for navigating the underlying large proof search spaces. While the existing approaches primarily rely on value functions and/or Monte Carlo Tree Search (MCTS), the potential of simpler methods like Best-First Tree Search (BFS) remains underexplored. In this paper, we investigate whether BFS can achieve competitive performance in large-scale theorem proving tasks. We present BFS-Prover, a scalable expert iteration framework, featuring three key innovations. First, we implement strategic data filtering at each expert iteration round, excluding problems solvable via beam search node expansion to focus on harder cases. Second, we improve the sample efficiency of BFS through Direct Preference Optimization (DPO) applied to state-tactic pairs automatically annotated with compiler error feedback, refining the LLM's policy to prioritize productive expansions. Third, we employ length normalization in BFS to encourage exploration of deeper proof paths. BFS-Prover achieves a state-of-the-art score of $72.95\%$ on the MiniF2F test set and therefore challenges the perceived necessity of complex tree search methods, demonstrating that BFS can achieve competitive performance when properly scaled. To facilitate further research and development in this area, we have open-sourced our model at https://huggingface.co/bytedance-research/BFS-Prover.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17026v1">Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Understanding the uncertainty in large language model (LLM) explanations is important for evaluating their faithfulness and reasoning consistency, and thus provides insights into the reliability of LLM's output regarding a question. In this work, we propose a novel framework that quantifies uncertainty in LLM explanations through a reasoning topology perspective. By designing a structural elicitation strategy, we guide the LLMs to frame the explanations of an answer into a graph topology. This process decomposes the explanations into the knowledge related sub-questions and topology-based reasoning structures, which allows us to quantify uncertainty not only at the semantic level but also from the reasoning path. It further brings convenience to assess knowledge redundancy and provide interpretable insights into the reasoning process. Our method offers a systematic way to interpret the LLM reasoning, analyze limitations, and provide guidance for enhancing robustness and faithfulness. This work pioneers the use of graph-structured uncertainty measurement in LLM explanations and demonstrates the potential of topology-based quantification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17011v1">Predicting Liquidity-Aware Bond Yields using Causal GANs and Deep Reinforcement Learning with LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Financial bond yield forecasting is challenging due to data scarcity, nonlinear macroeconomic dependencies, and evolving market conditions. In this paper, we propose a novel framework that leverages Causal Generative Adversarial Networks (CausalGANs) and Soft Actor-Critic (SAC) reinforcement learning (RL) to generate high-fidelity synthetic bond yield data for four major bond categories (AAA, BAA, US10Y, Junk). By incorporating 12 key macroeconomic variables, we ensure statistical fidelity by preserving essential market properties. To transform this market dependent synthetic data into actionable insights, we employ a finetuned Large Language Model (LLM) Qwen2.5-7B that generates trading signals (BUY/HOLD/SELL), risk assessments, and volatility projections. We use automated, human and LLM evaluations, all of which demonstrate that our framework improves forecasting performance over existing methods, with statistical validation via predictive accuracy, MAE evaluation(0.103%), profit/loss evaluation (60% profit rate), LLM evaluation (3.37/5) and expert assessments scoring 4.67 out of 5. The reinforcement learning-enhanced synthetic data generation achieves the least Mean Absolute Error of 0.103, demonstrating its effectiveness in replicating real-world bond market dynamics. We not only enhance data-driven trading strategies but also provides a scalable, high-fidelity synthetic financial data pipeline for risk & volatility management and investment decision-making. This work establishes a bridge between synthetic data generation, LLM driven financial forecasting, and language model evaluation, contributing to AI-driven financial decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10400v3">Reinforcement Learning Enhanced LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) enhanced large language models (LLMs), particularly exemplified by DeepSeek-R1, have exhibited outstanding performance. Despite the effectiveness in improving LLM capabilities, its implementation remains highly complex, requiring complex algorithms, reward modeling strategies, and optimization techniques. This complexity poses challenges for researchers and practitioners in developing a systematic understanding of RL-enhanced LLMs. Moreover, the absence of a comprehensive survey summarizing existing research on RL-enhanced LLMs has limited progress in this domain, hindering further advancements. In this work, we are going to make a systematic review of the most up-to-date state of knowledge on RL-enhanced LLMs, attempting to consolidate and analyze the rapidly growing research in this field, helping researchers understand the current challenges and advancements. Specifically, we (1) detail the basics of RL; (2) introduce popular RL-enhanced LLMs; (3) review researches on two widely-used reward model-based RL techniques: Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning from AI Feedback (RLAIF); and (4) explore Direct Preference Optimization (DPO), a set of methods that bypass the reward model to directly use human preference data for aligning LLM outputs with human expectations. We will also point out current challenges and deficiencies of existing methods and suggest some avenues for further improvements. Project page of this work can be found at https://github.com/ShuheWang1998/Reinforcement-Learning-Enhanced-LLMs-A-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11022v4">DynamicNER: A Dynamic, Multilingual, and Fine-Grained Dataset for LLM-based Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      With the advancement of Large Language Models (LLMs), more and more researchers apply LLMs for Named Entity Recognition (NER) methods, bringing vitality to this classical Natural Language Processing task. However, existing datasets are designed for traditional machine learning methods, inadequate for LLM-based methods in terms of corpus selection, entity categorization, and design logic. This limitation leads to less effective evaluation and model fine-tuning. To address this issue, we propose DynamicNER, the first NER dataset specifically designed for LLMs and with dynamic categorization, transcending the limitations of fixed categorization in existing datasets. It is also multi-lingual and multi-granular, covering 8 languages and 155 entity types, with corpus spanning multiple specialized domains. Furthermore, in response to the limitations demonstrated by existing LLM-based methods during DynamicNER testing, we develop CascadeNER, a novel NER method based on a two-stage strategy and lightweight LLMs, addressing the problems in current methods. Experiments show that DynamicNER is an effective benchmark for LLM-based NER methods, and CascadeNER outperforms existing methods with fewer computational resources. Our work is opened at https://github.com/CascadeNER/CascadeNER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16963v1">Make LLM Inference Affordable to Everyone: Augmenting GPU Memory with NDP-DIMM</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 15 pages, 17 figures, accepted by HPCA 2025
    </div>
    <details class="paper-abstract">
      The billion-scale Large Language Models (LLMs) need deployment on expensive server-grade GPUs with large-storage HBMs and abundant computation capability. As LLM-assisted services become popular, achieving cost-effective LLM inference on budget-friendly hardware becomes the trend. Extensive researches relocate LLM parameters from expensive GPUs to host memory. However, the restricted bandwidth between the host and GPU memory limits the inference performance. This work introduces Hermes, a budget-friendly system that leverages the near-data processing (NDP) within commodity DRAM DIMMs to enhance the performance of a single consumer-grade GPU, achieving efficient LLM inference. The inherent activation sparsity in LLMs naturally divides weight parameters into two categories, termed ``hot" and ``cold" neurons, respectively. Hot neurons, which consist of only approximately 20\% of all weight parameters, account for 80\% of the total computational load, while cold neurons make up the other 80\% of parameters but are responsible for just 20\% of the computational load. Therefore, we propose a heterogeneous computing strategy: mapping hot neurons to a single computation-efficient GPU, while offloading cold neurons to NDP-DIMMs, which offer large memory size but limited computation capabilities. Meanwhile, the dynamic nature of activation sparsity needs a real-time partition of hot/cold neurons and adaptive remapping of cold neurons across multiple NDP-DIMM modules. Therefore, we introduce a lightweight predictor optimizing real-time neuron partition and adjustment between GPU and NDP-DIMMs. We also utilize a window-based online scheduling mechanism to maintain load balance among NDP-DIMM modules. Hermes facilitates the deployment of LLaMA2-70B on consumer-grade hardware at 13.75 tokens/s and realizes an average 75.24$\times$ speedup over the state-of-the-art offloading-based inference system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15229v2">User Experience with LLM-powered Conversational Recommendation Systems: A Case of Music Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      The advancement of large language models (LLMs) now allows users to actively interact with conversational recommendation systems (CRS) and build their own personalized recommendation services tailored to their unique needs and goals. This experience offers users a significantly higher level of controllability compared to traditional RS, enabling an entirely new dimension of recommendation experiences. Building on this context, this study explored the unique experiences that LLM-powered CRS can provide compared to traditional RS. Through a three-week diary study with 12 participants using custom GPTs for music recommendations, we found that LLM-powered CRS can (1) help users clarify implicit needs, (2) support unique exploration, and (3) facilitate a deeper understanding of musical preferences. Based on these findings, we discuss the new design space enabled by LLM-powered CRS and highlight its potential to support more personalized, user-driven recommendation experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21271v3">EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      In this work, we re-formulate the model compression problem into the customized compensation problem: Given a compressed model, we aim to introduce residual low-rank paths to compensate for compression errors under customized requirements from users (e.g., tasks, compression ratios), resulting in greater flexibility in balancing accuracy and overhead(inference and model size) without being bound to fixed compression formats. However, naively applying SVD to derive residual paths causes suboptimal utilization of the low-rank representation capacity. Instead, we propose Training-free Eigenspace Low-Rank Approximation (EoRA), a method that directly minimizes compression-induced errors without requiring gradient-based training, achieving fast optimization in minutes using a small amount of calibration data. EoRA projects compression errors into the eigenspace of input activations, leveraging eigenvalues to effectively prioritize the reconstruction of high-importance error components. Moreover, EoRA can be seamlessly integrated with fine-tuning and quantization to further improve effectiveness and efficiency. EoRA consistently outperforms previous methods in compensating errors for compressed LLaMA2/3 models on various tasks, such as language generation, commonsense reasoning, and math reasoning tasks (e.g., 31.31%/12.88% and 9.69% improvements on ARC-Easy/ARC-Challenge and MathQA when compensating LLaMA3-8B that is quantized to 4-bit and pruned to 2:4 sparsity). EoRA offers a scalable, training-free solution to compensate for compression errors, making it a powerful tool to deploy LLMs more flexibly. Code is available at https://github.com/NVlabs/EoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16924v1">FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based cold-start recommendation systems continue to face significant computational challenges in billion-scale scenarios, as they follow a "Text-to-Judgment" paradigm. This approach processes user-item content pairs as input and evaluates each pair iteratively. To maintain efficiency, existing methods rely on pre-filtering a small candidate pool of user-item pairs. However, this severely limits the inferential capabilities of LLMs by reducing their scope to only a few hundred pre-filtered candidates. To overcome this limitation, we propose a novel "Text-to-Distribution" paradigm, which predicts an item's interaction probability distribution for the entire user set in a single inference. Specifically, we present FilterLLM, a framework that extends the next-word prediction capabilities of LLMs to billion-scale filtering tasks. FilterLLM first introduces a tailored distribution prediction and cold-start framework. Next, FilterLLM incorporates an efficient user-vocabulary structure to train and store the embeddings of billion-scale users. Finally, we detail the training objectives for both distribution prediction and user-vocabulary construction. The proposed framework has been deployed on the Alibaba platform, where it has been serving cold-start recommendations for two months, processing over one billion cold items. Extensive experiments demonstrate that FilterLLM significantly outperforms state-of-the-art methods in cold-start recommendation tasks, achieving over 30 times higher efficiency. Furthermore, an online A/B test validates its effectiveness in billion-scale recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16895v1">Unlocking Scientific Concepts: How Effective Are LLM-Generated Analogies for Student Understanding and Classroom Practice?</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 19 pages, conditionally accepted by CHI 2025
    </div>
    <details class="paper-abstract">
      Teaching scientific concepts is essential but challenging, and analogies help students connect new concepts to familiar ideas. Advancements in large language models (LLMs) enable generating analogies, yet their effectiveness in education remains underexplored. In this paper, we first conducted a two-stage study involving high school students and teachers to assess the effectiveness of LLM-generated analogies in biology and physics through a controlled in-class test and a classroom field study. Test results suggested that LLM-generated analogies could enhance student understanding particularly in biology, but require teachers' guidance to prevent over-reliance and overconfidence. Classroom experiments suggested that teachers could refine LLM-generated analogies to their satisfaction and inspire new analogies from generated ones, encouraged by positive classroom feedback and homework performance boosts. Based on findings, we developed and evaluated a practical system to help teachers generate and refine teaching analogies. We discussed future directions for developing and evaluating LLM-supported teaching and learning by analogy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12913v2">GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) fine-tuning technologies have achieved remarkable results. However, traditional LLM fine-tuning approaches face significant challenges: they require large Floating Point (FP) computation, raising privacy concerns when handling sensitive data, and are impractical for resource-constrained edge devices. While Parameter-Efficient Fine-Tuning (PEFT) techniques reduce trainable parameters, their reliance on floating-point arithmetic creates fundamental incompatibilities with edge hardware. In this work, we introduce a novel framework for on-device LLM fine-tuning that eliminates the need for floating-point operations in both inference and training, named GSQ-Tuning. At its core is the Group-Shared Exponents Integer format, which efficiently represents model parameters in integer format using shared exponents among parameter groups. When combined with LoRA-like adapters, this enables fully integer-based fine-tuning that is both memory and compute efficient. We demonstrate that our approach achieves accuracy comparable to BF16-based fine-tuning while significantly reducing 1.85x memory usage. Moreover, compared to FP8, our method can reduce 5x power consumption and 11x chip area with same performance, making large-scale model adaptation feasible on edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16892v1">Applying LLMs to Active Learning: Towards Cost-Efficient Cross-Task Text Classification without Manually Labeled Data</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Statement in Accordance with IEEE Preprint Policy: This work is intended for submission to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Machine learning-based classifiers have been used for text classification, such as sentiment analysis, news classification, and toxic comment classification. However, supervised machine learning models often require large amounts of labeled data for training, and manual annotation is both labor-intensive and requires domain-specific knowledge, leading to relatively high annotation costs. To address this issue, we propose an approach that integrates large language models (LLMs) into an active learning framework. Our approach combines the Robustly Optimized BERT Pretraining Approach (RoBERTa), Generative Pre-trained Transformer (GPT), and active learning, achieving high cross-task text classification performance without the need for any manually labeled data. Furthermore, compared to directly applying GPT for classification tasks, our approach retains over 93% of its classification performance while requiring only approximately 6% of the computational time and monetary cost, effectively balancing performance and resource efficiency. These findings provide new insights into the efficient utilization of LLMs and active learning algorithms in text classification tasks, paving the way for their broader application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09642v2">Krutrim LLM: Multilingual Foundational Model for over a Billion People</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      India is a diverse society with unique challenges in developing AI systems, including linguistic diversity, oral traditions, data accessibility, and scalability. Existing foundation models are primarily trained on English, limiting their effectiveness for India's population. Indic languages comprise only 1 percent of Common Crawl corpora despite India representing 18 percent of the global population, leading to linguistic biases. Thousands of regional languages, dialects, and code mixing create additional representation challenges due to sparse training data. We introduce Krutrim LLM, a 2 trillion token multilingual model designed for India's linguistic landscape. It incorporates the largest known Indic dataset, mitigating data scarcity and ensuring balanced performance across dialects. Krutrim outperforms or matches state-of-the-art models on Indic benchmarks while maintaining competitive English performance. Despite being significantly smaller in training flops, Krutrim LLM matches or exceeds models like LLAMA-2 on 10 out of 16 tasks, with an average score of 0.57 versus 0.55. This evidences Krutrim's flexible multilingual fluency across diverse linguistic contexts. Krutrim is integrated with real-time search to improve factual accuracy in conversational AI applications. This enhances accessibility for over 1 billion users worldwide. Through intentional design choices addressing data imbalances, Krutrim LLM signifies meaningful progress in building ethical, globally representative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16879v1">A Multi-LLM-Agent-Based Framework for Economic and Public Policy Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      This paper pioneers a novel approach to economic and public policy analysis by leveraging multiple Large Language Models (LLMs) as heterogeneous artificial economic agents. We first evaluate five LLMs' economic decision-making capabilities in solving two-period consumption allocation problems under two distinct scenarios: with explicit utility functions and based on intuitive reasoning. While previous research has often simulated heterogeneity by solely varying prompts, our approach harnesses the inherent variations in analytical capabilities across different LLMs to model agents with diverse cognitive traits. Building on these findings, we construct a Multi-LLM-Agent-Based (MLAB) framework by mapping these LLMs to specific educational groups and corresponding income brackets. Using interest-income taxation as a case study, we demonstrate how the MLAB framework can simulate policy impacts across heterogeneous agents, offering a promising new direction for economic and public policy analysis by leveraging LLMs' human-like reasoning capabilities and computational power.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10593v3">An Efficient Sign Language Translation Using Spatial Configuration and Motion Dynamics with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Accepted to NAACL 2025 main
    </div>
    <details class="paper-abstract">
      Gloss-free Sign Language Translation (SLT) converts sign videos directly into spoken language sentences without relying on glosses. Recently, Large Language Models (LLMs) have shown remarkable translation performance in gloss-free methods by harnessing their powerful natural language generation capabilities. However, these methods often rely on domain-specific fine-tuning of visual encoders to achieve optimal results. By contrast, this paper emphasizes the importance of capturing the spatial configurations and motion dynamics inherent in sign language. With this in mind, we introduce Spatial and Motion-based Sign Language Translation (SpaMo), a novel LLM-based SLT framework. The core idea of SpaMo is simple yet effective. We first extract spatial and motion features using off-the-shelf visual encoders and then input these features into an LLM with a language prompt. Additionally, we employ a visual-text alignment process as a warm-up before the SLT supervision. Our experiments demonstrate that SpaMo achieves state-of-the-art performance on two popular datasets, PHOENIX14T and How2Sign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13522v2">INDIC QA BENCHMARK: A Multilingual Benchmark to Evaluate Question Answering capability of LLMs for Indic Languages</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) perform well on unseen tasks in English, but their abilities in non English languages are less explored due to limited benchmarks and training data. To bridge this gap, we introduce the Indic QA Benchmark, a large dataset for context grounded question answering in 11 major Indian languages, covering both extractive and abstractive tasks. Evaluations of multilingual LLMs, including instruction finetuned versions, revealed weak performance in low resource languages due to a strong English language bias in their training data. We also investigated the Translate Test paradigm,where inputs are translated to English for processing and the results are translated back into the source language for output. This approach outperformed multilingual LLMs, particularly in low resource settings. By releasing Indic QA, we aim to promote further research into LLMs question answering capabilities in low resource languages. This benchmark offers a critical resource to address existing limitations and foster multilingual understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16852v1">Improving LLM General Preference Alignment via Optimistic Online Mirror Descent</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning from human feedback (RLHF) has demonstrated remarkable effectiveness in aligning large language models (LLMs) with human preferences. Many existing alignment approaches rely on the Bradley-Terry (BT) model assumption, which assumes the existence of a ground-truth reward for each prompt-response pair. However, this assumption can be overly restrictive when modeling complex human preferences. In this paper, we drop the BT model assumption and study LLM alignment under general preferences, formulated as a two-player game. Drawing on theoretical insights from learning in games, we integrate optimistic online mirror descent into our alignment framework to approximate the Nash policy. Theoretically, we demonstrate that our approach achieves an $O(T^{-1})$ bound on the duality gap, improving upon the previous $O(T^{-1/2})$ result. More importantly, we implement our method and show through experiments that it outperforms state-of-the-art RLHF algorithms across multiple representative benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14866v4">PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16362v2">AgentFL: Scaling LLM-based Fault Localization to Project-Level Context</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Added a comment to refer to the published version
    </div>
    <details class="paper-abstract">
      Fault Localization (FL) is an essential step during the debugging process. With the strong capabilities of code comprehension, the recent Large Language Models (LLMs) have demonstrated promising performance in diagnosing bugs in the code. Nevertheless, due to LLMs' limited performance in handling long contexts, existing LLM-based fault localization remains on localizing bugs within a small code scope (i.e., a method or a class), which struggles to diagnose bugs for a large code scope (i.e., an entire software system). To address the limitation, this paper presents AgentFL, a multi-agent system based on ChatGPT for automated fault localization. By simulating the behavior of a human developer, AgentFL models the FL task as a three-step process, which involves comprehension, navigation, and confirmation. Within each step, AgentFL hires agents with diversified expertise, each of which utilizes different tools to handle specific tasks. Particularly, we adopt a series of auxiliary strategies such as Test Behavior Tracking, Document-Guided Search, and Multi-Round Dialogue to overcome the challenges in each step. The evaluation on the widely used Defects4J-V1.2.0 benchmark shows that AgentFL can localize 157 out of 395 bugs within Top-1, which outperforms the other LLM-based approaches and exhibits complementarity to the state-of-the-art learning-based techniques. Additionally, we confirm the indispensability of the components in AgentFL with the ablation study and demonstrate the usability of AgentFL through a user study. Finally, the cost analysis shows that AgentFL spends an average of only 0.074 dollars and 97 seconds for a single bug.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16789v1">AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Alpha mining, a critical component in quantitative investment, focuses on discovering predictive signals for future asset returns in increasingly complex financial markets. However, the pervasive issue of alpha decay, where factors lose their predictive power over time, poses a significant challenge for alpha mining. Traditional methods like genetic programming face rapid alpha decay from overfitting and complexity, while approaches driven by Large Language Models (LLMs), despite their promise, often rely too heavily on existing knowledge, creating homogeneous factors that worsen crowding and accelerate decay. To address this challenge, we propose AlphaAgent, an autonomous framework that effectively integrates LLM agents with ad hoc regularizations for mining decay-resistant alpha factors. AlphaAgent employs three key mechanisms: (i) originality enforcement through a similarity measure based on abstract syntax trees (ASTs) against existing alphas, (ii) hypothesis-factor alignment via LLM-evaluated semantic consistency between market hypotheses and generated factors, and (iii) complexity control via AST-based structural constraints, preventing over-engineered constructions that are prone to overfitting. These mechanisms collectively guide the alpha generation process to balance originality, financial rationale, and adaptability to evolving market conditions, mitigating the risk of alpha decay. Extensive evaluations show that AlphaAgent outperforms traditional and LLM-based methods in mitigating alpha decay across bull and bear markets, consistently delivering significant alpha in Chinese CSI 500 and US S&P 500 markets over the past four years. Notably, AlphaAgent showcases remarkable resistance to alpha decay, elevating the potential for yielding powerful factors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16781v1">MultiOCR-QA: Dataset for Evaluating Robustness of LLMs in Question Answering on Multilingual OCR Texts</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Optical Character Recognition (OCR) plays a crucial role in digitizing historical and multilingual documents, yet OCR errors -- imperfect extraction of the text, including character insertion, deletion and permutation -- can significantly impact downstream tasks like question-answering (QA). In this work, we introduce a multilingual QA dataset MultiOCR-QA, designed to analyze the effects of OCR noise on QA systems' performance. The MultiOCR-QA dataset comprises 60K question-answer pairs covering three languages, English, French, and German. The dataset is curated from OCR-ed old documents, allowing for the evaluation of OCR-induced challenges on question answering. We evaluate MultiOCR-QA on various levels and types of OCR errors to access the robustness of LLMs in handling real-world digitization errors. Our findings show that QA systems are highly prone to OCR induced errors and exhibit performance degradation on noisy OCR text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10757v2">Deep Learning and LLM-based Methods Applied to Stellar Lightcurve Classification</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 35 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Light curves serve as a valuable source of information on stellar formation and evolution. With the rapid advancement of machine learning techniques, it can be effectively processed to extract astronomical patterns and information. In this study, we present a comprehensive evaluation of deep-learning and large language model (LLM) based models for the automatic classification of variable star light curves, based on large datasets from the Kepler and K2 missions. Special emphasis is placed on Cepheids, RR Lyrae, and eclipsing binaries, examining the influence of observational cadence and phase distribution on classification precision. Employing AutoDL optimization, we achieve striking performance with the 1D-Convolution+BiLSTM architecture and the Swin Transformer, hitting accuracies of 94\% and 99\% correspondingly, with the latter demonstrating a notable 83\% accuracy in discerning the elusive Type II Cepheids-comprising merely 0.02\% of the total dataset.We unveil StarWhisper LightCurve (LC), an innovative Series comprising three LLM-based models: LLM, multimodal large language model (MLLM), and Large Audio Language Model (LALM). Each model is fine-tuned with strategic prompt engineering and customized training methods to explore the emergent abilities of these models for astronomical data. Remarkably, StarWhisper LC Series exhibit high accuracies around 90\%, significantly reducing the need for explicit feature engineering, thereby paving the way for streamlined parallel data processing and the progression of multifaceted multimodal models in astronomical applications. The study furnishes two detailed catalogs illustrating the impacts of phase and sampling intervals on deep learning classification accuracy, showing that a substantial decrease of up to 14\% in observation duration and 21\% in sampling points can be realized without compromising accuracy by more than 10\%.
    </details>
</div>
