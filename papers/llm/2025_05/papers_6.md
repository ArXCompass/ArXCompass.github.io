# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11953v1">Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Loss reweighting has shown significant benefits for machine unlearning with large language models (LLMs). However, their exact functionalities are left unclear and the optimal strategy remains an open question, thus impeding the understanding and improvement of existing methodologies. In this paper, we identify two distinct goals of loss reweighting, namely, Saturation and Importance -- the former indicates that those insufficiently optimized data should be emphasized, while the latter stresses some critical data that are most influential for loss minimization. To study their usefulness, we design specific reweighting strategies for each goal and evaluate their respective effects on unlearning. We conduct extensive empirical analyses on well-established benchmarks, and summarize some important observations as follows: (i) Saturation enhances efficacy more than importance-based reweighting, and their combination can yield additional improvements. (ii) Saturation typically allocates lower weights to data with lower likelihoods, whereas importance-based reweighting does the opposite. (iii) The efficacy of unlearning is also largely influenced by the smoothness and granularity of the weight distributions. Based on these findings, we propose SatImp, a simple reweighting method that combines the advantages of both saturation and importance. Empirical results on extensive datasets validate the efficacy of our method, potentially bridging existing research gaps and indicating directions for future research. Our code is available at https://github.com/Puning97/SatImp-for-LLM-Unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11942v1">LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Lifelong learning is essential for intelligent agents operating in dynamic environments. Current large language model (LLM)-based agents, however, remain stateless and unable to accumulate or transfer knowledge over time. Existing benchmarks treat agents as static systems and fail to evaluate lifelong learning capabilities. We present LifelongAgentBench, the first unified benchmark designed to systematically assess the lifelong learning ability of LLM agents. It provides skill-grounded, interdependent tasks across three interactive environments, Database, Operating System, and Knowledge Graph, with automatic label verification, reproducibility, and modular extensibility. Extensive experiments reveal that conventional experience replay has limited effectiveness for LLM agents due to irrelevant information and context length constraints. We further introduce a group self-consistency mechanism that significantly improves lifelong learning performance. We hope LifelongAgentBench will advance the development of adaptive, memory-capable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11916v1">Arrow: Adaptive Scheduling Mechanisms for Disaggregated LLM Inference Architecture</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Existing large language models (LLMs) serving systems typically employ Prefill-Decode disaggregated architecture to prevent computational interference between the prefill and decode phases. However, real-world LLM serving scenarios often exhibit significant fluctuations in request input/output lengths, causing traditional static prefill/decode node configuration ratio to result in imbalanced computational loads between these two nodes, consequently preventing efficient utilization of computing resources to improve the system's goodput. To address this challenge, we design and implement Arrow, an adaptive scheduler that leverages stateless instances and elastic instance pools to achieve efficient adaptive request and instance scheduling. Arrow dynamically adjusts the number of instances handling prefill and decode tasks based on real-time cluster performance metrics, significantly enhancing the system's capability to handle traffic spikes and load variations. Our evaluation under diverse real-world workloads shows that Arrow achieves up to $5.62 \times$ and $7.78 \times$ higher request serving rates compared to state-of-the-art PD-colocated and PD-disaggregated serving systems respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11901v1">Benchmarking LLMs in an Embodied Environment for Blue Team Threat Hunting</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      As cyber threats continue to grow in scale and sophistication, blue team defenders increasingly require advanced tools to proactively detect and mitigate risks. Large Language Models (LLMs) offer promising capabilities for enhancing threat analysis. However, their effectiveness in real-world blue team threat-hunting scenarios remains insufficiently explored. In this paper, we present CYBERTEAM, a benchmark designed to guide LLMs in blue teaming practice. CYBERTEAM constructs an embodied environment in two stages. First, it models realistic threat-hunting workflows by capturing the dependencies among analytical tasks from threat attribution to incident response. Next, each task is addressed through a set of embodied functions tailored to its specific analytical requirements. This transforms the overall threat-hunting process into a structured sequence of function-driven operations, where each node represents a discrete function and edges define the execution order. Guided by this framework, LLMs are directed to perform threat-hunting tasks through modular steps. Overall, CYBERTEAM integrates 30 tasks and 9 embodied functions, guiding LLMs through pipelined threat analysis. We evaluate leading LLMs and state-of-the-art cybersecurity agents, comparing CYBERTEAM's embodied function-calling against fundamental elicitation strategies. Our results offer valuable insights into the current capabilities and limitations of LLMs in threat hunting, laying the foundation for the practical adoption in real-world cybersecurity applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11890v1">LLM-Enhanced Feature Engineering for Multi-Factor Electricity Price Predictions</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Accurately forecasting electricity price volatility is crucial for effective risk management and decision-making. Traditional forecasting models often fall short in capturing the complex, non-linear dynamics of electricity markets, particularly when external factors like weather conditions and market volatility are involved. These limitations hinder their ability to provide reliable predictions in markets with high volatility, such as the New South Wales (NSW) electricity market. To address these challenges, we introduce FAEP, a Feature-Augmented Electricity Price Prediction framework. FAEP leverages Large Language Models (LLMs) combined with advanced feature engineering to enhance prediction accuracy. By incorporating external features such as weather data and price volatility jumps, and utilizing Retrieval-Augmented Generation (RAG) for effective feature extraction, FAEP overcomes the shortcomings of traditional approaches. A hybrid XGBoost-LSTM model in FAEP further refines these augmented features, resulting in a more robust prediction framework. Experimental results demonstrate that FAEP achieves state-of-art (SOTA) performance compared to other electricity price prediction models in the Australian New South Wale electricity market, showcasing the efficiency of LLM-enhanced feature engineering and hybrid machine learning architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11888v1">AR Secretary Agent: Real-time Memory Augmentation via LLM-powered Augmented Reality Glasses</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Interacting with a significant number of individuals on a daily basis is commonplace for many professionals, which can lead to challenges in recalling specific details: Who is this person? What did we talk about last time? The advant of augmented reality (AR) glasses, equipped with visual and auditory data capture capabilities, presents a solution. In our work, we implemented an AR Secretary Agent with advanced Large Language Models (LLMs) and Computer Vision technologies. This system could discreetly provide real-time information to the wearer, identifying who they are conversing with and summarizing previous discussions. To verify AR Secretary, we conducted a user study with 13 participants and showed that our technique can efficiently help users to memorize events by up to 20\% memory enhancement on our study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02790v2">Leveraging the true depth of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable capabilities at the cost of high compute requirements. Recent studies have demonstrated that intermediate layers in LLMs can be removed or reordered without substantial accuracy loss; however, this insight has not yet been exploited to improve inference efficiency. Leveraging observed layer independence, we propose a novel method that groups consecutive layers into pairs evaluated in parallel, effectively restructuring the computational graph to enhance parallelism. Without requiring retraining or fine-tuning, this approach achieves an inference throughput improvement of 1.05x-1.20x on standard benchmarks, retaining 95\%-99\% of the original model accuracy. Empirical results demonstrate the practicality of this method in significantly reducing inference cost for large-scale LLM deployment. Additionally, we demonstrate that modest performance degradation can be substantially mitigated through lightweight fine-tuning, further enhancing the method's applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12162v2">AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Modern large language model (LLM) applications exhibit diverse service-level objectives (SLOs), from low-latency requirements in interactive coding assistants to more relaxed constraints in data wrangling tasks. Existing LLM serving systems, which rely on uniform batching and scheduling strategies, often fail to meet these heterogeneous SLOs concurrently. We present AdaServe, the first LLM serving system designed to support efficient multi-SLO serving through SLO-customized speculative decoding. AdaServe formulates multi-SLO serving as a constrained optimization problem and introduces a hardware-aware algorithm that constructs a speculation tree tailored to each request's latency target. It features a speculate-select-verify pipeline that enables fine-grained control over decoding speed while maximizing system throughput. AdaServe further adapts to workload variation by dynamically adjusting speculation parameters. Evaluations across diverse workloads show that AdaServe reduces SLO violations by up to 4.3$\times$ and improves goodput by up to 1.9$\times$ compared to the best performing baselines, highlighting its effectiveness in multi-SLO serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11875v1">J1: Exploring Simple Test-Time Scaling for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 33 pages, 27 figures
    </div>
    <details class="paper-abstract">
      The current focus of AI research is shifting from emphasizing model training towards enhancing evaluation quality, a transition that is crucial for driving further advancements in AI systems. Traditional evaluation methods typically rely on reward models assigning scalar preference scores to outputs. Although effective, such approaches lack interpretability, leaving users often uncertain about why a reward model rates a particular response as high or low. The advent of LLM-as-a-Judge provides a more scalable and interpretable method of supervision, offering insights into the decision-making process. Moreover, with the emergence of large reasoning models, which consume more tokens for deeper thinking and answer refinement, scaling test-time computation in the LLM-as-a-Judge paradigm presents an avenue for further boosting performance and providing more interpretability through reasoning traces. In this paper, we introduce $\textbf{J1-7B}$, which is first supervised fine-tuned on reflection-enhanced datasets collected via rejection-sampling and subsequently trained using Reinforcement Learning (RL) with verifiable rewards. At inference time, we apply Simple Test-Time Scaling (STTS) strategies for additional performance improvement. Experimental results demonstrate that $\textbf{J1-7B}$ surpasses the previous state-of-the-art LLM-as-a-Judge by $ \textbf{4.8}$\% and exhibits a $ \textbf{5.1}$\% stronger scaling trend under STTS. Additionally, we present three key findings: (1) Existing LLM-as-a-Judge does not inherently exhibit such scaling trend. (2) Model simply fine-tuned on reflection-enhanced datasets continues to demonstrate similarly weak scaling behavior. (3) Significant scaling trend emerges primarily during the RL phase, suggesting that effective STTS capability is acquired predominantly through RL training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11861v1">Fair-PP: A Synthetic Dataset for Aligning LLM with Personalized Preferences of Social Equity</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Human preference plays a crucial role in the refinement of large language models (LLMs). However, collecting human preference feedback is costly and most existing datasets neglect the correlation between personalization and preferences. To address this issue, we introduce Fair-PP, a synthetic dataset of personalized preferences targeting social equity, derived from real-world social survey data, which includes 28 social groups, 98 equity topics, and 5 personal preference dimensions. Leveraging GPT-4o-mini, we engage in role-playing based on seven representative persona portrayals guided by existing social survey data, yielding a total of 238,623 preference records. Through Fair-PP, we also contribute (i) An automated framework for generating preference data, along with a more fine-grained dataset of personalized preferences; (ii) analysis of the positioning of the existing mainstream LLMs across five major global regions within the personalized preference space; and (iii) a sample reweighting method for personalized preference alignment, enabling alignment with a target persona while maximizing the divergence from other personas. Empirical experiments show our method outperforms the baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06964v2">Bridging AI and Carbon Capture: A Dataset for LLMs in Ionic Liquids and CBE Research</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance in general knowledge and reasoning tasks across various domains. However, their effectiveness in specialized scientific fields like Chemical and Biological Engineering (CBE) remains underexplored. Addressing this gap requires robust evaluation benchmarks that assess both knowledge and reasoning capabilities in these niche areas, which are currently lacking. To bridge this divide, we present a comprehensive empirical analysis of LLM reasoning capabilities in CBE, with a focus on Ionic Liquids (ILs) for carbon sequestration - an emerging solution for mitigating global warming. We develop and release an expert - curated dataset of 5,920 examples designed to benchmark LLMs' reasoning in this domain. The dataset incorporates varying levels of difficulty, balancing linguistic complexity and domain-specific knowledge. Using this dataset, we evaluate three open-source LLMs with fewer than 10 billion parameters. Our findings reveal that while smaller general-purpose LLMs exhibit basic knowledge of ILs, they lack the specialized reasoning skills necessary for advanced applications. Building on these results, we discuss strategies to enhance the utility of LLMs for carbon capture research, particularly using ILs. Given the significant carbon footprint of LLMs, aligning their development with IL research presents a unique opportunity to foster mutual progress in both fields and advance global efforts toward achieving carbon neutrality by 2050.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11839v1">On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      Counterfactual reasoning has emerged as a crucial technique for generalizing the reasoning capabilities of large language models (LLMs). By generating and analyzing counterfactual scenarios, researchers can assess the adaptability and reliability of model decision-making. Although prior work has shown that LLMs often struggle with counterfactual reasoning, it remains unclear which factors most significantly impede their performance across different tasks and modalities. In this paper, we propose a decompositional strategy that breaks down the counterfactual generation from causality construction to the reasoning over counterfactual interventions. To support decompositional analysis, we investigate 11 datasets spanning diverse tasks, including natural language understanding, mathematics, programming, and vision-language tasks. Through extensive evaluations, we characterize LLM behavior across each decompositional stage and identify how modality type and intermediate reasoning influence performance. By establishing a structured framework for analyzing counterfactual reasoning, this work contributes to the development of more reliable LLM-based reasoning systems and informs future elicitation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18280v3">Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      The security issue of large language models (LLMs) has gained wide attention recently, with various defense mechanisms developed to prevent harmful output, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the output distribution of text embedding models is severely biased with a large mean. Inspired by this observation, we propose novel, efficient methods to search for **universal magic words** that attack text embedding models. Universal magic words as suffixes can shift the embedding of any text towards the bias direction, thus manipulating the similarity of any text pair and misleading safeguards. Attackers can jailbreak the safeguards by appending magic words to user prompts and requiring LLMs to end answers with magic words. Experiments show that magic word attacks significantly degrade safeguard performance on JailbreakBench, cause real-world chatbots to produce harmful outputs in full-pipeline attacks, and generalize across input/output texts, models, and languages. To eradicate this security risk, we also propose defense methods against such attacks, which can correct the bias of text embeddings and improve downstream performance in a train-free manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11827v1">Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 In progress
    </div>
    <details class="paper-abstract">
      Compressing long chain-of-thought (CoT) from large language models (LLMs) is an emerging strategy to improve the reasoning efficiency of LLMs. Despite its promising benefits, existing studies equally compress all thoughts within a long CoT, hindering more concise and effective reasoning. To this end, we first investigate the importance of different thoughts by examining their effectiveness and efficiency in contributing to reasoning through automatic long CoT chunking and Monte Carlo rollouts. Building upon the insights, we propose a theoretically bounded metric to jointly measure the effectiveness and efficiency of different thoughts. We then propose Long$\otimes$Short, an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts. Specifically, we begin by synthesizing a small amount of cold-start data to fine-tune LLMs for long-thought and short-thought reasoning styles, respectively. Furthermore, we propose a synergizing-oriented multi-turn reinforcement learning, focusing on the model self-evolution and collaboration between long-thought and short-thought LLMs. Experimental results show that our method enables Qwen2.5-7B and Llama3.1-8B to achieve comparable performance compared to DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B, while reducing token length by over 80% across the MATH500, AIME24/25, AMC23, and GPQA Diamond benchmarks. Our data and code are available at https://github.com/yasNing/Long-otimes-Short/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11449v1">LLMs unlock new paths to monetizing exploits</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      We argue that Large language models (LLMs) will soon alter the economics of cyberattacks. Instead of attacking the most commonly used software and monetizing exploits by targeting the lowest common denominator among victims, LLMs enable adversaries to launch tailored attacks on a user-by-user basis. On the exploitation front, instead of human attackers manually searching for one difficult-to-identify bug in a product with millions of users, LLMs can find thousands of easy-to-identify bugs in products with thousands of users. And on the monetization front, instead of generic ransomware that always performs the same attack (encrypt all your data and request payment to decrypt), an LLM-driven ransomware attack could tailor the ransom demand based on the particular content of each exploited device. We show that these two attacks (and several others) are imminently practical using state-of-the-art LLMs. For example, we show that without any human intervention, an LLM finds highly sensitive personal information in the Enron email dataset (e.g., an executive having an affair with another employee) that could be used for blackmail. While some of our attacks are still too expensive to scale widely today, the incentives to implement these attacks will only increase as LLMs get cheaper. Thus, we argue that LLMs create a need for new defense-in-depth approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11413v1">CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in medical contexts, raising critical concerns about safety, alignment, and susceptibility to adversarial manipulation. While prior benchmarks assess model refusal capabilities for harmful prompts, they often lack clinical specificity, graded harmfulness levels, and coverage of jailbreak-style attacks. We introduce CARES (Clinical Adversarial Robustness and Evaluation of Safety), a benchmark for evaluating LLM safety in healthcare. CARES includes over 18,000 prompts spanning eight medical safety principles, four harm levels, and four prompting styles: direct, indirect, obfuscated, and role-play, to simulate both malicious and benign use cases. We propose a three-way response evaluation protocol (Accept, Caution, Refuse) and a fine-grained Safety Score metric to assess model behavior. Our analysis reveals that many state-of-the-art LLMs remain vulnerable to jailbreaks that subtly rephrase harmful prompts, while also over-refusing safe but atypically phrased queries. Finally, we propose a mitigation strategy using a lightweight classifier to detect jailbreak attempts and steer models toward safer behavior via reminder-based conditioning. CARES provides a rigorous framework for testing and improving medical LLM safety under adversarial and ambiguous conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11401v1">Can AI automatically analyze public opinion? A LLM agents-based agentic pipeline for timely public opinion analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 43 pages, 3 figures, 4 tables (1 in appendix), includes appendix. Preprint only
    </div>
    <details class="paper-abstract">
      This study proposes and implements the first LLM agents based agentic pipeline for multi task public opinion analysis. Unlike traditional methods, it offers an end-to-end, fully automated analytical workflow without requiring domain specific training data, manual annotation, or local deployment. The pipeline integrates advanced LLM capabilities into a low-cost, user-friendly framework suitable for resource constrained environments. It enables timely, integrated public opinion analysis through a single natural language query, making it accessible to non-expert users. To validate its effectiveness, the pipeline was applied to a real world case study of the 2025 U.S. China tariff dispute, where it analyzed 1,572 Weibo posts and generated a structured, multi part analytical report. The results demonstrate some relationships between public opinion and governmental decision-making. These contributions represent a novel advancement in applying generative AI to public governance, bridging the gap between technical sophistication and practical usability in public opinion monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02406v4">Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. In this paper, we answer the following question: Given a piece of text, can we identify whether it was produced by a particular LLM or not? We model LLM-generated text as a sequential stochastic process with complete dependence on history. We then design zero-shot statistical tests to (i) distinguish between text generated by two different known sets of LLMs $A$ (non-sanctioned) and $B$ (in-house), and (ii) identify whether text was generated by a known LLM or generated by any unknown model, e.g., a human or some other language generation process. We prove that the type I and type II errors of our test decrease exponentially with the length of the text. For that, we show that if $B$ generates the text, then except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. We then present experiments using LLMs with white-box access to support our theoretical results and empirically examine the robustness of our results to black-box settings and adversarial attacks. In the black-box setting, our method achieves an average TPR of 82.5\% at a fixed FPR of 5\%. Under adversarial perturbations, our minimum TPR is 48.6\% at the same FPR threshold. Both results outperform all non-commercial baselines. See https://github.com/TaraRadvand74/llm-text-detection for code, data, and an online demo of the project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14662v2">iAgent: LLM Agent as a Shield between User and Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Findings of ACL 2025 and WWW2025@HCRS
    </div>
    <details class="paper-abstract">
      Traditional recommender systems usually take the user-platform paradigm, where users are directly exposed under the control of the platform's recommendation algorithms. However, the defect of recommendation algorithms may put users in very vulnerable positions under this paradigm. First, many sophisticated models are often designed with commercial objectives in mind, focusing on the platform's benefits, which may hinder their ability to protect and capture users' true interests. Second, these models are typically optimized using data from all users, which may overlook individual user's preferences. Due to these shortcomings, users may experience several disadvantages under the traditional user-platform direct exposure paradigm, such as lack of control over the recommender system, potential manipulation by the platform, echo chamber effects, or lack of personalization for less active users due to the dominance of active users during collaborative learning. Therefore, there is an urgent need to develop a new paradigm to protect user interests and alleviate these issues. Recently, some researchers have introduced LLM agents to simulate user behaviors, these approaches primarily aim to optimize platform-side performance, leaving core issues in recommender systems unresolved. To address these limitations, we propose a new user-agent-platform paradigm, where agent serves as the protective shield between user and recommender system that enables indirect exposure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13837v2">Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 30 pages, 27 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11368v1">GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed as autonomous agents capable of following user instructions and making decisions in real-world applications. Previous studies have made notable progress in benchmarking the instruction following capabilities of LLMs in general domains, with a primary focus on their inherent commonsense knowledge. Recently, LLMs have been increasingly deployed as domain-oriented agents, which rely on domain-oriented guidelines that may conflict with their commonsense knowledge. These guidelines exhibit two key characteristics: they consist of a wide range of domain-oriented rules and are subject to frequent updates. Despite these challenges, the absence of comprehensive benchmarks for evaluating the domain-oriented guideline following capabilities of LLMs presents a significant obstacle to their effective assessment and further development. In this paper, we introduce GuideBench, a comprehensive benchmark designed to evaluate guideline following performance of LLMs. GuideBench evaluates LLMs on three critical aspects: (i) adherence to diverse rules, (ii) robustness to rule updates, and (iii) alignment with human preferences. Experimental results on a range of LLMs indicate substantial opportunities for improving their ability to follow domain-oriented guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17773v2">Uncertainty Quantification for LLM-Based Survey Simulations</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 33 pages, 7 figures, 10 tables
    </div>
    <details class="paper-abstract">
      We investigate the use of large language models (LLMs) to simulate human responses to survey questions, and perform uncertainty quantification to gain reliable insights. Our approach converts imperfect LLM-simulated responses into confidence sets for population parameters of human responses, addressing the distribution shift between the simulated and real populations. A key innovation lies in determining the optimal number of simulated responses: too many produce overly narrow confidence sets with poor coverage, while too few yield excessively loose estimates. To resolve this, our method adaptively selects the simulation sample size, ensuring valid average-case coverage guarantees. It is broadly applicable to any LLM, irrespective of its fidelity, and any procedure for constructing confidence sets. Additionally, the selected sample size quantifies the degree of misalignment between the LLM and the target human population. We illustrate our method on real datasets and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11352v1">LegoSLM: Connecting LLM with Speech Encoder using CTC Posteriors</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Recently, large-scale pre-trained speech encoders and Large Language Models (LLMs) have been released, which show state-of-the-art performance on a range of spoken language processing tasks including Automatic Speech Recognition (ASR). To effectively combine both models for better performance, continuous speech prompts, and ASR error correction have been adopted. However, these methods are prone to suboptimal performance or are inflexible. In this paper, we propose a new paradigm, LegoSLM, that bridges speech encoders and LLMs using the ASR posterior matrices. The speech encoder is trained to generate Connectionist Temporal Classification (CTC) posteriors over the LLM vocabulary, which are used to reconstruct pseudo-audio embeddings by computing a weighted sum of the LLM input embeddings. These embeddings are concatenated with text embeddings in the LLM input space. Using the well-performing USM and Gemma models as an example, we demonstrate that our proposed LegoSLM method yields good performance on both ASR and speech translation tasks. By connecting USM with Gemma models, we can get an average of 49% WERR over the USM-CTC baseline on 8 MLS testsets. The trained model also exhibits modularity in a range of settings -- after fine-tuning the Gemma model weights, the speech encoder can be switched and combined with the LLM in a zero-shot fashion. Additionally, we propose to control the decode-time influence of the USM and LLM using a softmax temperature, which shows effectiveness in domain adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01503v5">A Highly Scalable LLM Clusters with Optical Interconnect</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      We propose \emph{LumosCore} to build high-bandwidth and large-scale data center networks for LLM jobs. By replacing the core-layer electrical packet switches by optical circuit switches, \emph{LumosCore} could achieves $2\times$ increase in bandwidth or $8\times$ increase in network size. We offer the detailed design of \emph{LumosCore} at both deployment stage and running stage. At deployment stage, we propose Cross Wiring, which is compatible with all possible logical topologies. At running stage, we design polynomial-time algorithms for GPU placement, logical topology generating and OCS reconfiguration to minimize network contention and reduce impact to scheduled jobs. We evaluate \emph{LumosCore} using both testbed experiments and large-scale simulation. Compared to traditional hybrid optical/electrical architectures, \emph{LumosCore} increases the end-to-end training throughput by up to 39.5\% on a 128-node testbed. Compared to the state-of-art Clos architectures, \emph{LumosCore} reduces the average job completion time by up to 34.1\% in a 16k simulation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11336v1">XtraGPT: LLMs for Human-AI Collaboration on Controllable Academic Paper Revision</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Despite the growing adoption of large language models (LLMs) in academic workflows, their capabilities remain limited when it comes to supporting high-quality scientific writing. Most existing systems are designed for general-purpose scientific text generation and fail to meet the sophisticated demands of research communication beyond surface-level polishing, such as conceptual coherence across sections. Furthermore, academic writing is inherently iterative and revision-driven, a process not well supported by direct prompting-based paradigms. To address these scenarios, we propose a human-AI collaboration framework for academic paper revision. We first introduce a comprehensive dataset of 7,040 research papers from top-tier venues annotated with over 140,000 instruction-response pairs that reflect realistic, section-level scientific revisions. Building on the dataset, we develop XtraGPT, the first suite of open-source LLMs, designed to provide context-aware, instruction-guided writing assistance, ranging from 1.5B to 14B parameters. Extensive experiments validate that XtraGPT significantly outperforms same-scale baselines and approaches the quality of proprietary systems. Both automated preference assessments and human evaluations confirm the effectiveness of our models in improving scientific drafts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16466v3">On the Feasibility of Using LLMs to Autonomously Execute Multi-host Network Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 18 pages, 15 figures
    </div>
    <details class="paper-abstract">
      LLMs have shown preliminary promise in some security tasks and CTF challenges. Real cyberattacks are often multi-host network attacks, which involve executing a number of steps across multiple hosts such as conducting reconnaissance, exploiting vulnerabilities, and using compromised hosts to exfiltrate data. To date, the extent to which LLMs can autonomously execute multi-host network attacks} is not well understood. To this end, our first contribution is MHBench, an open-source multi-host attack benchmark with 10 realistic emulated networks (from 25 to 50 hosts). We find that popular LLMs including modern reasoning models (e.g., GPT4o, Gemini 2.5 Pro, Sonnet 3.7 Thinking) with state-of-art security-relevant prompting strategies (e.g., PentestGPT, CyberSecEval3) cannot autonomously execute multi-host network attacks. To enable LLMs to autonomously execute such attacks, our second contribution is Incalmo, an high-level abstraction layer. Incalmo enables LLMs to specify high-level actions (e.g., infect a host, scan a network). Incalmo's translation layer converts these actions into lower-level primitives (e.g., commands to exploit tools) through expert agents. In 9 out of 10 networks in MHBench, LLMs using Incalmo achieve at least some of the attack goals. Even smaller LLMs (e.g., Haiku 3.5, Gemini 2 Flash) equipped with Incalmo achieve all goals in 5 of 10 environments. We also validate the key role of high-level actions in Incalmo's abstraction in enabling LLMs to autonomously execute such attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11329v1">TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 13 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Distributed inference of large language models (LLMs) can introduce overheads of up to 20% even over GPUs connected via high-speed interconnects such as NVLINK. Multiple techniques have been proposed to mitigate these overheads by decomposing computations into finer-grained tasks and overlapping communication with sub-tasks as they complete. However, fine-grained decomposition of a large computation into many smaller computations on GPUs results in overheads. Further, the communication itself uses many streaming multiprocessors (SMs), adding to the overhead. We present TokenWeave to address these challenges. TokenWeave proposes a Token-Splitting technique that divides the tokens in the inference batch into two approximately equal subsets in a wave-aware manner. The computation of one subset is then overlapped with the communication of the other. In addition, TokenWeave optimizes the order of the layer normalization computation with respect to communication operations and implements a novel fused AllReduce-RMSNorm kernel carefully leveraging Multimem instruction support available on NVIDIA Hopper GPUs. These optimizations allow TokenWeave to perform communication and RMSNorm using only 2-8 SMs. Moreover, our kernel enables the memory bound RMSNorm to be overlapped with the other batch's computation, providing additional gains. Our evaluations demonstrate up to 29% latency gains and up to 26% throughput gains across multiple models and workloads. In several settings, TokenWeave results in better performance compared to an equivalent model with all communication removed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11277v1">Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04588v2">ZeroSearch: Incentivize the Search Capability of LLMs without Searching</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a novel RL framework that incentivizes the capabilities of LLMs to use a real search engine with simulated searches during training. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both useful and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11247v1">LD-Scene: LLM-Guided Diffusion for Controllable Generation of Adversarial Safety-Critical Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Ensuring the safety and robustness of autonomous driving systems necessitates a comprehensive evaluation in safety-critical scenarios. However, these safety-critical scenarios are rare and difficult to collect from real-world driving data, posing significant challenges to effectively assessing the performance of autonomous vehicles. Typical existing methods often suffer from limited controllability and lack user-friendliness, as extensive expert knowledge is essentially required. To address these challenges, we propose LD-Scene, a novel framework that integrates Large Language Models (LLMs) with Latent Diffusion Models (LDMs) for user-controllable adversarial scenario generation through natural language. Our approach comprises an LDM that captures realistic driving trajectory distributions and an LLM-based guidance module that translates user queries into adversarial loss functions, facilitating the generation of scenarios aligned with user queries. The guidance module integrates an LLM-based Chain-of-Thought (CoT) code generator and an LLM-based code debugger, enhancing the controllability and robustness in generating guidance functions. Extensive experiments conducted on the nuScenes dataset demonstrate that LD-Scene achieves state-of-the-art performance in generating realistic, diverse, and effective adversarial scenarios. Furthermore, our framework provides fine-grained control over adversarial behaviors, thereby facilitating more effective testing tailored to specific driving scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11227v1">Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      The development of reasoning capabilities represents a critical frontier in large language models (LLMs) research, where reinforcement learning (RL) and process reward models (PRMs) have emerged as predominant methodological frameworks. Contrary to conventional wisdom, empirical evidence from DeepSeek-R1 demonstrates that pure RL training focused on mathematical problem-solving can progressively enhance reasoning abilities without PRM integration, challenging the perceived necessity of process supervision. In this study, we conduct a systematic investigation of the relationship between RL training and PRM capabilities. Our findings demonstrate that problem-solving proficiency and process supervision capabilities represent complementary dimensions of reasoning that co-evolve synergistically during pure RL training. In particular, current PRMs underperform simple baselines like majority voting when applied to state-of-the-art models such as DeepSeek-R1 and QwQ-32B. To address this limitation, we propose Self-PRM, an introspective framework in which models autonomously evaluate and rerank their generated solutions through self-reward mechanisms. Although Self-PRM consistently improves the accuracy of the benchmark (particularly with larger sample sizes), analysis exposes persistent challenges: The approach exhibits low precision (<10\%) on difficult problems, frequently misclassifying flawed solutions as valid. These analyses underscore the need for continued RL scaling to improve reward alignment and introspective accuracy. Overall, our findings suggest that PRM may not be essential for enhancing complex reasoning, as pure RL not only improves problem-solving skills but also inherently fosters robust PRM capabilities. We hope these findings provide actionable insights for building more reliable and self-aware complex reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08881v2">Can We Trust AI Agents? A Case Study of an LLM-Based Multi-Agent System for Ethical AI</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      AI-based systems, including Large Language Models (LLM), impact millions by supporting diverse tasks but face issues like misinformation, bias, and misuse. AI ethics is crucial as new technologies and concerns emerge, but objective, practical guidance remains debated. This study examines the use of LLMs for AI ethics in practice, assessing how LLM trustworthiness-enhancing techniques affect software development in this context. Using the Design Science Research (DSR) method, we identify techniques for LLM trustworthiness: multi-agents, distinct roles, structured communication, and multiple rounds of debate. We design a multi-agent prototype LLM-MAS, where agents engage in structured discussions on real-world AI ethics issues from the AI Incident Database. We evaluate the prototype across three case scenarios using thematic analysis, hierarchical clustering, comparative (baseline) studies, and running source code. The system generates approximately 2,000 lines of code per case, compared to only 80 lines in baseline trials. Discussions reveal terms like bias detection, transparency, accountability, user consent, GDPR compliance, fairness evaluation, and EU AI Act compliance, showing this prototype ability to generate extensive source code and documentation addressing often overlooked AI ethics issues. However, practical challenges in source code integration and dependency management may limit its use by practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11189v1">Can Global XAI Methods Reveal Injected Bias in LLMs? SHAP vs Rule Extraction vs RuleSHAP</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Generative AI systems can help spread information but also misinformation and biases, potentially undermining the UN Sustainable Development Goals (SDGs). Explainable AI (XAI) aims to reveal the inner workings of AI systems and expose misbehaviours or biases. However, current XAI tools, built for simpler models, struggle to handle the non-numerical nature of large language models (LLMs). This paper examines the effectiveness of global XAI methods, such as rule-extraction algorithms and SHAP, in detecting bias in LLMs. To do so, we first show a text-to-ordinal mapping strategy to convert non-numerical inputs/outputs into numerical features, enabling these tools to identify (some) misinformation-related biases in LLM-generated content. Then, we inject non-linear biases of varying complexity (univariate, conjunctive, and non-convex) into widespread LLMs like ChatGPT and Llama via system instructions, using global XAI methods to detect them. This way, we found that RuleFit struggles with conjunctive and non-convex biases, while SHAP can approximate conjunctive biases but cannot express them as actionable rules. Hence, we introduce RuleSHAP, a global rule extraction algorithm combining SHAP and RuleFit to detect more non-univariate biases, improving injected bias detection over RuleFit by +94% (MRR@1) on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16525v2">KVShare: An LLM Service System with Efficient and Effective Multi-Tenant KV Cache Reuse</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Recent advances in long-text understanding have pushed the context length of large language models (LLMs) up to one million tokens. It boosts LLMs's accuracy and reasoning capacity but causes exorbitant computational costs and unsatisfactory Time to First Token (TTFT). KV cache reuse, which reuses the exact same KV cache of prefixes and templates or shares similar ones but with extra selective recomputation, offers a promising way to tackle this issue. However, prior studies overlook the cross-request KV reuse and the attention deviations introduced by new tokens during the decoding stage. In this paper, we present a KV cache management module that shares the KV cache across requests under multi-tenant scenarios without sacrificing model accuracy. Our system, KVShare, enables accurate and efficient LLM serving by 1) a Dual-Stage High Deviation algorithm (DHD) that conditionally selects a small portion of KV cache to be recomputed during both prefill and decode phases, and 2) a cache-aware scheduler that prioritizes requests based on their KV cache hit rates and orchestrates continuous batching to achieve enhanced system efficiency and faster TTFT. Multi-task experiments conducted on models such as Qwen2.5-7B,Llama3.1-8B and Yi1.5-9B demonstrate that KVShare reduces TTFT by up to 9.39x and increases 1.2x of the throughput compared to the full KV recompute. Moreover, KVShare achieves 20.38% boost in terms of accuracy compared to SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11183v1">On Next-Token Prediction in LLMs: How End Goals Determine the Consistency of Decoding Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Probabilistic next-token prediction trained using cross-entropy loss is the basis of most large language models. Given a sequence of previous values, next-token prediction assigns a probability to each possible next value in the vocabulary. There are many ways to use next-token prediction to output token sequences. This paper examines a few of these algorithms (greedy, lookahead, random sampling, and temperature-scaled random sampling) and studies their consistency with respect to various goals encoded as loss functions. Although consistency of surrogate losses with respect to a target loss function is a well researched topic, we are the first to study it in the context of LLMs (to the best of our knowledge). We find that, so long as next-token prediction converges to its true probability distribution, random sampling is consistent with outputting sequences that mimic sampling from the true probability distribution. For the other goals, such as minimizing the 0-1 loss on the entire sequence, we show no polynomial-time algorithm is optimal for all probability distributions and all decoding algorithms studied are only optimal for a subset of probability distributions. When analyzing these results, we see that there is a dichotomy created between the goals of information retrieval and creative generation for the decoding algorithms. This shows that choosing the correct decoding algorithm based on the desired goal is extremely important and many of the ones used are lacking theoretical grounding in numerous scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11166v1">SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Despite advances in pretraining with extended context lengths, large language models (LLMs) still face challenges in effectively utilizing real-world long-context information, primarily due to insufficient long-context alignment caused by data quality issues, training inefficiencies, and the lack of well-designed optimization objectives. To address these limitations, we propose a framework named $\textbf{S}$h$\textbf{o}$rt-to-$\textbf{Lo}$ng $\textbf{P}$reference $\textbf{O}$ptimization ($\textbf{SoLoPO}$), decoupling long-context preference optimization (PO) into two components: short-context PO and short-to-long reward alignment (SoLo-RA), supported by both theoretical and empirical evidence. Specifically, short-context PO leverages preference pairs sampled from short contexts to enhance the model's contextual knowledge utilization ability. Meanwhile, SoLo-RA explicitly encourages reward score consistency utilization for the responses when conditioned on both short and long contexts that contain identical task-relevant information. This facilitates transferring the model's ability to handle short contexts into long-context scenarios. SoLoPO is compatible with mainstream preference optimization algorithms, while substantially improving the efficiency of data construction and training processes. Experimental results show that SoLoPO enhances all these algorithms with respect to stronger length and domain generalization abilities across various long-context benchmarks, while achieving notable improvements in both computational and memory efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16116v2">DMind Benchmark: Toward a Holistic Assessment of LLM Capabilities across the Web3 Domain</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved impressive performance in diverse natural language processing tasks, but specialized domains such as Web3 present new challenges and require more tailored evaluation. Despite the significant user base and capital flows in Web3, encompassing smart contracts, decentralized finance (DeFi), non-fungible tokens (NFTs), decentralized autonomous organizations (DAOs), on-chain governance, and novel token-economics, no comprehensive benchmark has systematically assessed LLM performance in this domain. To address this gap, we introduce the DMind Benchmark, a holistic Web3-oriented evaluation suite covering nine critical subfields: fundamental blockchain concepts, blockchain infrastructure, smart contract, DeFi mechanisms, DAOs, NFTs, token economics, meme concept, and security vulnerabilities. Beyond multiple-choice questions, DMind Benchmark features domain-specific tasks such as contract debugging and on-chain numeric reasoning, mirroring real-world scenarios. We evaluated 26 models, including ChatGPT, Claude, DeepSeek, Gemini, Grok, and Qwen, uncovering notable performance gaps in specialized areas like token economics and security-critical contract analysis. While some models excel in blockchain infrastructure tasks, advanced subfields remain challenging. Our benchmark dataset and evaluation pipeline are open-sourced on https://huggingface.co/datasets/DMindAI/DMind_Benchmark, reaching number one in Hugging Face's trending dataset charts within a week of release.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09724v2">An AI-Powered Research Assistant in the Lab: A Practical Guide for Text Analysis Through Iterative Collaboration with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 31 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Analyzing texts such as open-ended responses, headlines, or social media posts is a time- and labor-intensive process highly susceptible to bias. LLMs are promising tools for text analysis, using either a predefined (top-down) or a data-driven (bottom-up) taxonomy, without sacrificing quality. Here we present a step-by-step tutorial to efficiently develop, test, and apply taxonomies for analyzing unstructured data through an iterative and collaborative process between researchers and LLMs. Using personal goals provided by participants as an example, we demonstrate how to write prompts to review datasets and generate a taxonomy of life domains, evaluate and refine the taxonomy through prompt and direct modifications, test the taxonomy and assess intercoder agreements, and apply the taxonomy to categorize an entire dataset with high intercoder reliability. We discuss the possibilities and limitations of using LLMs for text analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11122v1">Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for Formulaic Factor Mining</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Alpha factor mining is pivotal in quantitative investment for identifying predictive signals from complex financial data. While traditional formulaic alpha mining relies on human expertise, contemporary automated methods, such as those based on genetic programming or reinforcement learning, often suffer from search inefficiency or yield poorly interpretable alpha factors. This paper introduces a novel framework that integrates Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS) to overcome these limitations. Our approach leverages the LLM's instruction-following and reasoning capability to iteratively generate and refine symbolic alpha formulas within an MCTS-driven exploration. A key innovation is the guidance of MCTS exploration by rich, quantitative feedback from financial backtesting of each candidate factor, enabling efficient navigation of the vast search space. Furthermore, a frequent subtree avoidance mechanism is introduced to bolster search efficiency and alpha factor performance. Experimental results on real-world stock market data demonstrate that our LLM-based framework outperforms existing methods by mining alphas with superior predictive accuracy, trading performance, and improved interpretability, while offering a more efficient solution for formulaic alpha mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06581v2">HAFLQ: Heterogeneous Adaptive Federated LoRA Fine-tuned LLM with Quantization</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 This is an extended journal version based on our previous conference paper accepted at the 2025 IEEE International Conference on Communications (ICC), with additional sections and new results
    </div>
    <details class="paper-abstract">
      Federated fine-tuning of pre-trained Large Language Models (LLMs) enables task-specific adaptation across diverse datasets while preserving privacy. However, challenges such as high computational and memory demands, heterogeneous client resources, bandwidth constraints, and ineffective global aggregation hinder its efficiency. To address these issues, we propose HAFLQ (Heterogeneous Adaptive Federated Low-Rank Adaptation Fine-tuned LLM with Quantization), a novel framework for efficient and scalable federated fine-tuning of LLMs in heterogeneous environments. To reduce memory and computation demands, we propose a salience-driven adaptive LLM quantization framework that evaluates the importance of transformer blocks using a salience metric and applies adaptive block-wise quantization accordingly. To handle heterogeneous computational capabilities, we propose an importance-based parameter truncation and freezing scheme. To address communication bottlenecks, we propose an importance-aware bandwidth-adaptive quantization method, which dynamically adjusts parameter precision based on importance and bandwidth constraints. To improve global model aggregation, we propose an adaptive rank-1 matrix-level aggregation strategy, which prevents information dilution and accelerates convergence by aggregating only updated rank-1 matrices from clients. Experimental results on the text classification task demonstrate that HAFLQ reduces memory usage by 31%, lowers communication cost by 49%, improves accuracy by 50%, and achieves faster convergence compared to the baseline method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11081v1">ShiQ: Bringing back Bellman to LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      The fine-tuning of pre-trained large language models (LLMs) using reinforcement learning (RL) is generally formulated as direct policy optimization. This approach was naturally favored as it efficiently improves a pretrained LLM, seen as an initial policy. Another RL paradigm, Q-learning methods, has received far less attention in the LLM community while demonstrating major success in various non-LLM RL tasks. In particular, Q-learning effectiveness comes from its sample efficiency and ability to learn offline, which is particularly valuable given the high computational cost of sampling with LLMs. However, naively applying a Q-learning-style update to the model's logits is ineffective due to the specificity of LLMs. Our core contribution is to derive theoretically grounded loss functions from Bellman equations to adapt Q-learning methods to LLMs. To do so, we carefully adapt insights from the RL literature to account for LLM-specific characteristics, ensuring that the logits become reliable Q-value estimates. We then use this loss to build a practical algorithm, ShiQ for Shifted-Q, that supports off-policy, token-wise learning while remaining simple to implement. Finally, we evaluate ShiQ on both synthetic data and real-world benchmarks, e.g., UltraFeedback and BFCL-V3, demonstrating its effectiveness in both single-turn and multi-turn LLM settings
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11077v1">LLM-Enhanced Symbolic Control for Safety-Critical Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Motivated by Smart Manufacturing and Industry 4.0, we introduce a framework for synthesizing Abstraction-Based Controller Design (ABCD) for reach-avoid problems from Natural Language (NL) specifications using Large Language Models (LLMs). A Code Agent interprets an NL description of the control problem and translates it into a formal language interpretable by state-of-the-art symbolic control software, while a Checker Agent verifies the correctness of the generated code and enhances safety by identifying specification mismatches. Evaluations show that the system handles linguistic variability and improves robustness over direct planning with LLMs. The proposed approach lowers the barrier to formal control synthesis by enabling intuitive, NL-based task definition while maintaining safety guarantees through automated validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00031v2">Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Machine-learning technologies are seeing increased deployment in real-world market scenarios. In this work, we explore the strategic behaviors of large language models (LLMs) when deployed as autonomous agents in multi-commodity markets, specifically within Cournot competition frameworks. We examine whether LLMs can independently engage in anti-competitive practices such as collusion or, more specifically, market division. Our findings demonstrate that LLMs can effectively monopolize specific commodities by dynamically adjusting their pricing and resource allocation strategies, thereby maximizing profitability without direct human input or explicit collusion commands. These results pose unique challenges and opportunities for businesses looking to integrate AI into strategic roles and for regulatory bodies tasked with maintaining fair and competitive markets. The study provides a foundation for further exploration into the ramifications of deferring high-stakes decisions to LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09921v2">PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Accepted to ACL 2025 (main)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at https://github.com/redwyd/PrivacyJailbreak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11011v1">Humans expect rationality and cooperation from LLM opponents in strategic games</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) integrate into our social and economic interactions, we need to deepen our understanding of how humans respond to LLMs opponents in strategic settings. We present the results of the first controlled monetarily-incentivised laboratory experiment looking at differences in human behaviour in a multi-player p-beauty contest against other humans and LLMs. We use a within-subject design in order to compare behaviour at the individual level. We show that, in this environment, human subjects choose significantly lower numbers when playing against LLMs than humans, which is mainly driven by the increased prevalence of `zero' Nash-equilibrium choices. This shift is mainly driven by subjects with high strategic reasoning ability. Subjects who play the zero Nash-equilibrium choice motivate their strategy by appealing to perceived LLM's reasoning ability and, unexpectedly, propensity towards cooperation. Our findings provide foundational insights into the multi-player human-LLM interaction in simultaneous choice games, uncover heterogeneities in both subjects' behaviour and beliefs about LLM's play when playing against them, and suggest important implications for mechanism design in mixed human-LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10981v1">Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 ACL 2025 Main
    </div>
    <details class="paper-abstract">
      Recently, scaling test-time compute on Large Language Models (LLM) has garnered wide attention. However, there has been limited investigation of how various reasoning prompting strategies perform as scaling. In this paper, we focus on a standard and realistic scaling setting: majority voting. We systematically conduct experiments on 6 LLMs $\times$ 8 prompting strategies $\times$ 6 benchmarks. Experiment results consistently show that as the sampling time and computational overhead increase, complicated prompting strategies with superior initial performance gradually fall behind simple Chain-of-Thought. We analyze this phenomenon and provide theoretical proofs. Additionally, we propose a method according to probability theory to quickly and accurately predict the scaling performance and select the best strategy under large sampling times without extra resource-intensive inference in practice. It can serve as the test-time scaling law for majority voting. Furthermore, we introduce two ways derived from our theoretical analysis to significantly improve the scaling performance. We hope that our research can promote to re-examine the role of complicated prompting, unleash the potential of simple prompting strategies, and provide new insights for enhancing test-time scaling performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10978v1">Group-in-Group Policy Optimization for LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in group-based reinforcement learning (RL) have driven frontier large language models (LLMs) in single-turn tasks like mathematical reasoning. However, their scalability to long-horizon LLM agent training remains limited. Unlike static tasks, agent-environment interactions unfold over many steps and often yield sparse or delayed rewards, making credit assignment across individual steps significantly more challenging. In this work, we propose Group-in-Group Policy Optimization (GiGPO), a novel RL algorithm that achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence. GiGPO introduces a two-level structure for estimating relative advantage: (i) At the episode-level, GiGPO computes macro relative advantages based on groups of complete trajectories; (ii) At the step-level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories. Actions stemming from the same state are grouped together, enabling micro relative advantage estimation. This hierarchical structure effectively captures both global trajectory quality and local step effectiveness without relying on auxiliary models or additional rollouts. We evaluate GiGPO on two challenging agent benchmarks, ALFWorld and WebShop, using Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct. Crucially, GiGPO delivers fine-grained per-step credit signals and achieves performance gains of > 12\% on ALFWorld and > 9\% on WebShop over the GRPO baseline: all while maintaining the same GPU memory overhead, identical LLM rollout, and incurring little to no additional time cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10948v1">The Way We Prompt: Conceptual Blending, Neural Dynamics, and Prompt-Induced Transitions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), inspired by neuroscience, exhibit behaviors that often evoke a sense of personality and intelligence-yet the mechanisms behind these effects remain elusive. Here, we operationalize Conceptual Blending Theory (CBT) as an experimental framework, using prompt-based methods to reveal how LLMs blend and compress meaning. By systematically investigating Prompt-Induced Transitions (PIT) and Prompt-Induced Hallucinations (PIH), we uncover structural parallels and divergences between artificial and biological cognition. Our approach bridges linguistics, neuroscience, and empirical AI research, demonstrating that human-AI collaboration can serve as a living prototype for the future of cognitive science. This work proposes prompt engineering not just as a technical tool, but as a scientific method for probing the deep structure of meaning itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10940v1">Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Recommender systems filter contents/items valuable to users by inferring preferences from user features and historical behaviors. Mainstream approaches follow the learning-to-rank paradigm, which focus on discovering and modeling item topics (e.g., categories), and capturing user preferences on these topics based on historical interactions. However, this paradigm often neglects the modeling of user characteristics and their social roles, which are logical confounders influencing the correlated interest and user preference transition. To bridge this gap, we introduce the user role identification task and the behavioral logic modeling task that aim to explicitly model user roles and learn the logical relations between item topics and user social roles. We show that it is possible to explicitly solve these tasks through an efficient integration framework of Large Language Model (LLM) and recommendation systems, for which we propose TagCF. On the one hand, the exploitation of the LLM's world knowledge and logic inference ability produces a virtual logic graph that reveals dynamic and expressive knowledge of users, augmenting the recommendation performance. On the other hand, the user role aligns the user behavioral logic with the observed user feedback, refining our understanding of user behaviors. Additionally, we also show that the extracted user-item logic graph is empirically a general knowledge that can benefit a wide range of recommendation tasks, and conduct experiments on industrial and several public datasets as verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10939v1">GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Accepted to ACL 2025 (main conference, short paper), 10 pages
    </div>
    <details class="paper-abstract">
      Large language models often struggle with zero-shot generalization, and several modular approaches have been proposed to address this challenge. Yet, we hypothesize that a key limitation remains: the entanglement of general knowledge and task-specific adaptations. To overcome this, we propose a modular framework that disentangles these components by constructing a library of task-specific LoRA modules alongside a general-domain LoRA. By subtracting this general knowledge component from each task-specific module, we obtain residual modules that focus more exclusively on task-relevant information, a method we call general knowledge subtraction (GenKnowSub). Leveraging the refined task-specific modules and the Arrow routing algorithm \citep{ostapenko2024towards}, we dynamically select and combine modules for new inputs without additional training. Our studies on the Phi-3 model and standard Arrow as baselines reveal that using general knowledge LoRAs derived from diverse languages, including English, French, and German, yields consistent performance gains in both monolingual and cross-lingual settings across a wide set of benchmarks. Further experiments on Phi-2 demonstrate how GenKnowSub generalizes to weaker LLMs. The complete code and data are available at https://github.com/saharsamr/Modular-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10936v1">Connecting the Dots: A Chain-of-Collaboration Prompting Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 34 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance in executing complex reasoning tasks. Chain-of-thought effectively enhances reasoning capabilities by unlocking the potential of large models, while multi-agent systems provide more comprehensive solutions by integrating collective intelligence of multiple agents. However, both approaches face significant limitations. Single-agent with chain-of-thought, due to the inherent complexity of designing cross-domain prompts, faces collaboration challenges. Meanwhile, multi-agent systems consume substantial tokens and inevitably dilute the primary problem, which is particularly problematic in business workflow tasks. To address these challenges, we propose Cochain, a collaboration prompting framework that effectively solves business workflow collaboration problem by combining knowledge and prompts at a reduced cost. Specifically, we construct an integrated knowledge graph that incorporates knowledge from multiple stages. Furthermore, by maintaining and retrieving a prompts tree, we can obtain prompt information relevant to other stages of the business workflow. We perform extensive evaluations of Cochain across multiple datasets, demonstrating that Cochain outperforms all baselines in both prompt engineering and multi-agent LLMs. Additionally, expert evaluation results indicate that the use of a small model in combination with Cochain outperforms GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11865v2">From Image to Video, what do we need in multimodal LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Covering from Image LLMs to the more complex Video LLMs, the Multimodal Large Language Models (MLLMs) have demonstrated profound capabilities in comprehending cross-modal information as numerous studies have illustrated. Previous methods delve into designing comprehensive Video LLMs through integrating video foundation models with primitive LLMs. Despite its effectiveness, such paradigm renders Video LLM's structure verbose and typically requires substantial video data for pre-training. Crucially, it neglects leveraging the foundational contributions of ready-made Image LLMs. In this paper, we introduce RED-VILLM, a Resource-Efficient Development pipeline which builds robust Video LLMs through leveraging the prior knowledge of Image LLMs. Specifically, since a video is naturally a combination of images along the temporal dimension, we devise a temporal adaptation plug-and-play structure, endowing the backbone Image LLM with the capability to grasp temporal information. Moreover, through applying this pipeline, we achieve the first Video LLM within the Chinese-speaking community. Extensive experiments demonstrate that Video LLMs developed through our approach surpass conventional Video LLMs, requiring minimal instructional data and training resources. Our approach highlights the potential for a more cost-effective and scalable advancement in multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09933v4">MIR-Bench: Can Your LLM Recognize Complicated Patterns via Many-Shot In-Context Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 36 pages, 11 figures. The last version adds more experiments and modifies name for better summary of the work
    </div>
    <details class="paper-abstract">
      The ability to recognize patterns from examples and apply them to new ones is a primal ability for general intelligence, and is widely studied by psychology and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually <10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations often focus on classification, and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context reasoning benchmark for pattern recognition that asks LLM to predict output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for many-shot in-context reasoning, and acquired many insightful findings including scaling effect, robustness, inductive vs. transductive reasoning, retrieval Augmented Generation (RAG), coding for inductive reasoning, cross-domain generalizability, etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10900v1">Explain What You Mean: Intent Augmented Knowledge Graph Recommender Built With LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Interaction sparsity is the primary obstacle for recommendation systems. Sparsity manifests in environments with disproportional cardinality of groupings of entities, such as users and products in an online marketplace. It also is found for newly introduced entities, described as the cold-start problem. Recent efforts to mitigate this sparsity issue shifts the performance bottleneck to other areas in the computational pipeline. Those that focus on enriching sparse representations with connectivity data from other external sources propose methods that are resource demanding and require careful domain expert aided addition of this newly introduced data. Others that turn to Large Language Model (LLM) based recommenders will quickly encounter limitations surrounding data quality and availability. In this work, we propose LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that leverages retrieval-augmented generation and an encoding approach to construct and densify a knowledge graph. IKGR learns latent user-item affinities from an interaction knowledge graph and further densifies it through mutual intent connectivity. This addresses sparsity issues and allows the model to make intent-grounded recommendations with an interpretable embedding translation layer. Through extensive experiments on real-world datasets, we demonstrate that IKGR overcomes knowledge gaps and achieves substantial gains over state-of-the-art baselines on both publicly available and our internal recommendation datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11507v4">TestAgent: A Framework for Domain-Adaptive Evaluation of LLMs via Dynamic Benchmark Construction and Exploratory Interaction</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed to various vertical domains, automatically evaluating their performance across different domains remains a critical challenge. Current evaluation methods often rely on static and resource-intensive datasets that are not aligned with real-world requirements and lack cross-domain adaptability. To address these limitations, we revisit the evaluation process and introduce two key concepts: \textbf{Benchmark+}, which extends the traditional question-answer benchmark into a more flexible ``strategy-criterion'' format; and \textbf{Assessment+}, which enhances the interaction process to facilitate deeper exploration and comprehensive analysis from multiple perspectives. We propose \textbf{\textsc{TestAgent}}, an agent-based evaluation framework that implements these concepts using retrieval-augmented generation and reinforcement learning. \textsc{TestAgent} enables automatic dynamic benchmark generation and in-depth assessment across diverse vertical domains. Experiments on tasks ranging from constructing multiple vertical domain evaluations to transforming static benchmarks into dynamic forms demonstrate the effectiveness of \textsc{TestAgent}. This work provides a novel perspective on automatic evaluation methods for domain-specific LLMs, offering a pathway for domain-adaptive dynamic benchmark construction and exploratory assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06326v5">Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle to provide up-to-date information due to their one-time training and the constantly evolving nature of the world. To keep LLMs current, existing approaches typically involve continued pre-training on new documents. However, they frequently face difficulties in extracting stored knowledge. Motivated by the remarkable success of the Feynman Technique in efficient human learning, we introduce Self-Tuning, a learning framework aimed at improving an LLM's ability to effectively acquire new knowledge from unseen raw documents through self-teaching. Specifically, we develop a Self-Teaching strategy that augments the documents with a set of knowledge-intensive tasks created in a self-supervised manner, focusing on three crucial aspects: memorization, comprehension, and self-reflection. Additionally, we introduce three Wiki-Newpages-2023-QA datasets to facilitate an in-depth analysis of an LLM's knowledge acquisition ability concerning memorization, extraction, and reasoning. Extensive experimental results on various models, e.g., Llama2-7B reveal that Self-Tuning consistently exhibits superior performance across all knowledge acquisition tasks and excels in preserving previous knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10861v1">Improving the Data-efficiency of Reinforcement Learning by Warm-starting with LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 31 pages (9 for the main paper), 27 figures, NeurIPS 25 submission
    </div>
    <details class="paper-abstract">
      We investigate the usage of Large Language Model (LLM) in collecting high-quality data to warm-start Reinforcement Learning (RL) algorithms for learning in some classical Markov Decision Process (MDP) environments. In this work, we focus on using LLM to generate an off-policy dataset that sufficiently covers state-actions visited by optimal policies, then later using an RL algorithm to explore the environment and improve the policy suggested by the LLM. Our algorithm, LORO, can both converge to an optimal policy and have a high sample efficiency thanks to the LLM's good starting policy. On multiple OpenAI Gym environments, such as CartPole and Pendulum, we empirically demonstrate that LORO outperforms baseline algorithms such as pure LLM-based policies, pure RL, and a naive combination of the two, achieving up to $4 \times$ the cumulative rewards of the pure RL baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07623v6">The Curious Case of Class Accuracy Imbalance in LLMs: Post-hoc Debiasing via Nonlinear Integer Programming</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are good knowledge bases but struggle to perform equally well for all classes in text classification. This paper investigates the case of class accuracy imbalance in LLMs, where deeply entangled pretraining biases and prompt-specific cues contribute to the imbalance. To overcome the difficulty in bias identification and inaccessibility of retraining, we post-hoc balance class accuracy using only output probabilities. This is enabled by reformulating debiasing as a combinatorial optimization problem. In details, we first motivate a post-hoc bias metric, the Contextual Oddity Bias (COBias), to quantify the over-/under-prediction (a tendency to over-predict some classes while under-predicting others) in LLMs. We then propose the Debiasing as Nonlinear Integer Programming (DNIP) method to reweight LLM output class probabilities towards minimizing COBias and max12 imizing overall accuracy, without being constrained by bias sources or updating LLM parameters. Since the DNIP model contains non-differentiable elements, we use simulated annealing to efficiently solve it. Evaluations on five LLMs across NLP classification benchmarks show that DNIP simultaneously achieves signifi16 cant COBias reduction (61% relative reduction) and accuracy improvement (18% relative increase) under different LLM prompting setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10838v1">LARGO: Latent Adversarial Reflection through Gradient Optimization for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Efficient red-teaming method to uncover vulnerabilities in Large Language Models (LLMs) is crucial. While recent attacks often use LLMs as optimizers, the discrete language space make gradient-based methods struggle. We introduce LARGO (Latent Adversarial Reflection through Gradient Optimization), a novel latent self-reflection attack that reasserts the power of gradient-based optimization for generating fluent jailbreaking prompts. By operating within the LLM's continuous latent space, LARGO first optimizes an adversarial latent vector and then recursively call the same LLM to decode the latent into natural language. This methodology yields a fast, effective, and transferable attack that produces fluent and stealthy prompts. On standard benchmarks like AdvBench and JailbreakBench, LARGO surpasses leading jailbreaking techniques, including AutoDAN, by 44 points in attack success rate. Our findings demonstrate a potent alternative to agentic LLM prompting, highlighting the efficacy of interpreting and attacking LLM internals through gradient optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v1">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. We open source our code at \href{https://github.com/uiuctml/MergeBench}{https://github.com/uiuctml/MergeBench}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10829v1">Enhancing Low-Resource Minority Language Translation with LLMs and Retrieval-Augmented Generation for Cultural Nuances</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Accepted to IntelliSys 2025
    </div>
    <details class="paper-abstract">
      This study investigates the challenges of translating low-resource languages by integrating Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG). Various model configurations were tested on Hakka translations, with BLEU scores ranging from 12% (dictionary-only) to 31% (RAG with Gemini 2.0). The best-performing model (Model 4) combined retrieval and advanced language modeling, improving lexical coverage, particularly for specialized or culturally nuanced terms, and enhancing grammatical coherence. A two-stage method (Model 3) using dictionary outputs refined by Gemini 2.0 achieved a BLEU score of 26%, highlighting iterative correction's value and the challenges of domain-specific expressions. Static dictionary-based approaches struggled with context-sensitive content, demonstrating the limitations of relying solely on predefined resources. These results emphasize the need for curated resources, domain knowledge, and ethical collaboration with local communities, offering a framework that improves translation accuracy and fluency while supporting cultural preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09665v2">Tales of the 2025 Los Angeles Fire: Hotwash for Public Health Concerns in Reddit via LLM-Enhanced Topic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Corrected capitalization errors in the section subtitle 3.4, 4.3, step 1 in section 3.3.2, and Supplementary Information. Fix typo with "Weighting" for step 4 in section 3.3.2
    </div>
    <details class="paper-abstract">
      Wildfires have become increasingly frequent, irregular, and severe in recent years. Understanding how affected populations perceive and respond during wildfire crises is critical for timely and empathetic disaster response. Social media platforms offer a crowd-sourced channel to capture evolving public discourse, providing hyperlocal information and insight into public sentiment. This study analyzes Reddit discourse during the 2025 Los Angeles wildfires, spanning from the onset of the disaster to full containment. We collect 385 posts and 114,879 comments related to the Palisades and Eaton fires. We adopt topic modeling methods to identify the latent topics, enhanced by large language models (LLMs) and human-in-the-loop (HITL) refinement. Furthermore, we develop a hierarchical framework to categorize latent topics, consisting of two main categories, Situational Awareness (SA) and Crisis Narratives (CN). The volume of SA category closely aligns with real-world fire progressions, peaking within the first 2-5 days as the fires reach the maximum extent. The most frequent co-occurring category set of public health and safety, loss and damage, and emergency resources expands on a wide range of health-related latent topics, including environmental health, occupational health, and one health. Grief signals and mental health risks consistently accounted for 60 percentage and 40 percentage of CN instances, respectively, with the highest total volume occurring at night. This study contributes the first annotated social media dataset on the 2025 LA fires, and introduces a scalable multi-layer framework that leverages topic modeling for crisis discourse analysis. By identifying persistent public health concerns, our results can inform more empathetic and adaptive strategies for disaster response, public health communication, and future research in comparable climate-related disaster events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11885v4">Med-R$^2$: Crafting Trustworthy LLM Physicians via Retrieval and Reasoning of Evidence-Based Medicine</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable capabilities in clinical scenarios. Despite their potential, existing works face challenges when applying LLMs to medical settings. Strategies relying on training with medical datasets are highly cost-intensive and may suffer from outdated training data. Leveraging external knowledge bases is a suitable alternative, yet it faces obstacles such as limited retrieval precision and poor effectiveness in answer extraction. These issues collectively prevent LLMs from demonstrating the expected level of proficiency in mastering medical expertise. To address these challenges, we introduce Med-R^2, a novel LLM physician framework that adheres to the Evidence-Based Medicine (EBM) process, efficiently integrating retrieval mechanisms as well as the selection and reasoning processes of evidence, thereby enhancing the problem-solving capabilities of LLMs in healthcare scenarios and fostering a trustworthy LLM physician. Our comprehensive experiments indicate that Med-R^2 achieves a 14.74\% improvement over vanilla RAG methods and even a 3.32\% enhancement compared to fine-tuning strategies, without incurring additional training costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12632v2">What External Knowledge is Preferred by LLMs? Characterizing and Exploring Chain of Evidence in Imperfect Context</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Incorporating external knowledge into large language models (LLMs) has emerged as a promising approach to mitigate outdated knowledge and hallucination in LLMs. However, external knowledge is often imperfect. In addition to useful knowledge, external knowledge is rich in irrelevant or misinformation in the context that can impair the reliability of LLM responses. This paper focuses on LLMs' preferred external knowledge in imperfect contexts when handling multi-hop QA. Inspired by criminal procedural law's Chain of Evidence (CoE), we characterize that knowledge preferred by LLMs should maintain both relevance to the question and mutual support among knowledge pieces. Accordingly, we propose an automated CoE discrimination approach and evaluate LLMs' effectiveness, faithfulness and robustness with CoE, including its application in the Retrieval-Augmented Generation (RAG). Tests on five LLMs show CoE improves generation accuracy, answer faithfulness, robustness to knowledge conflicts, and boosts the performance of existing approaches in three practical RAG scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10782v1">EdgeMM: Multi-Core CPU with Heterogeneous AI-Extension and Activation-aware Weight Pruning for Multimodal LLMs at Edge</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Accepted by DAC 2025
    </div>
    <details class="paper-abstract">
      Emerging multimodal LLMs (MLLMs) exhibit strong cross-modality perception and reasoning capabilities and hold great potential for various applications at edge. However, MLLMs typically consist of a compute-intensive modality encoder and a memory-bound LLM decoder, leading to distinct bottlenecks for hardware designs. In this work, we present a multi-core CPU solution with heterogeneous AI extensions, which are based on either the compute-centric systolic array or memory-centric digital compute-in-memory (CIM) co-processors. In addition, dynamic activation-aware weight pruning and bandwidth management are developed to enhance bandwidth efficiency and core utilization, improving overall performance. We implemented our solution using commercial 22nm technology. For representative MLLMs, our evaluations show EdgeMM can achieve 2.84x performance speedup compared to laptop 3060 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03049v2">34 Examples of LLM Applications in Materials Science and Chemistry: Towards Automation, Assistants, Agents, and Accelerated Scientific Discovery</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.15221. This paper is a refinement and analysis of the raw project submissions from arXiv:2411.15221
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping many aspects of materials science and chemistry research, enabling advances in molecular property prediction, materials design, scientific automation, knowledge extraction, and more. Recent developments demonstrate that the latest class of models are able to integrate structured and unstructured data, assist in hypothesis generation, and streamline research workflows. To explore the frontier of LLM capabilities across the research lifecycle, we review applications of LLMs through 34 total projects developed during the second annual Large Language Model Hackathon for Applications in Materials Science and Chemistry, a global hybrid event. These projects spanned seven key research areas: (1) molecular and material property prediction, (2) molecular and material design, (3) automation and novel interfaces, (4) scientific communication and education, (5) research data management and automation, (6) hypothesis generation and evaluation, and (7) knowledge extraction and reasoning from the scientific literature. Collectively, these applications illustrate how LLMs serve as versatile predictive models, platforms for rapid prototyping of domain-specific tools, and much more. In particular, improvements in both open source and proprietary LLM performance through the addition of reasoning, additional training data, and new techniques have expanded effectiveness, particularly in low-data environments and interdisciplinary research. As LLMs continue to improve, their integration into scientific workflows presents both new opportunities and new challenges, requiring ongoing exploration, continued refinement, and further research to address reliability, interpretability, and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03266v2">LLM Content Moderation and User Satisfaction: Evidence from Response Refusals in Chatbot Arena</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      LLM safety and ethical alignment are widely discussed, but the impact of content moderation on user satisfaction remains underexplored. In particular, little is known about how users respond when models refuse to answer a prompt-one of the primary mechanisms used to enforce ethical boundaries in LLMs. We address this gap by analyzing nearly 50,000 model comparisons from Chatbot Arena, a platform where users indicate their preferred LLM response in pairwise matchups, providing a large-scale setting for studying real-world user preferences. Using a novel RoBERTa-based refusal classifier fine-tuned on a hand-labeled dataset, we distinguish between refusals due to ethical concerns and technical limitations. Our results reveal a substantial refusal penalty: ethical refusals yield significantly lower win rates than both technical refusals and standard responses, indicating that users are especially dissatisfied when models decline a task for ethical reasons. However, this penalty is not uniform. Refusals receive more favorable evaluations when the underlying prompt is highly sensitive (e.g., involving illegal content), and when the refusal is phrased in a detailed and contextually aligned manner. These findings underscore a core tension in LLM design: safety-aligned behaviors may conflict with user expectations, calling for more adaptive moderation strategies that account for context and presentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10774v1">Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Time series forecasting is important for applications spanning energy markets, climate analysis, and traffic management. However, existing methods struggle to effectively integrate exogenous texts and align them with the probabilistic nature of large language models (LLMs). Current approaches either employ shallow text-time series fusion via basic prompts or rely on deterministic numerical decoding that conflict with LLMs' token-generation paradigm, which limits contextual awareness and distribution modeling. To address these limitations, we propose CAPTime, a context-aware probabilistic multimodal time series forecasting method that leverages text-informed abstraction and autoregressive LLM decoding. Our method first encodes temporal patterns using a pretrained time series encoder, then aligns them with textual contexts via learnable interactions to produce joint multimodal representations. By combining a mixture of distribution experts with frozen LLMs, we enable context-aware probabilistic forecasting while preserving LLMs' inherent distribution modeling capabilities. Experiments on diverse time series forecasting tasks demonstrate the superior accuracy and generalization of CAPTime, particularly in multimodal scenarios. Additional analysis highlights its robustness in data-scarce scenarios through hybrid probabilistic decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11718v1">REMOR: Automated Peer Review Generation with LLM Reasoning and Multi-Objective Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 18 pages, 6 figures
    </div>
    <details class="paper-abstract">
      AI-based peer review systems tend to produce shallow and overpraising suggestions compared to human feedback. Here, we evaluate how well a reasoning LLM trained with multi-objective reinforcement learning (REMOR) can overcome these limitations. We start by designing a multi-aspect reward function that aligns with human evaluation of reviews. The aspects are related to the review itself (e.g., criticisms, novelty) and the relationship between the review and the manuscript (i.e., relevance). First, we perform supervised fine-tuning of DeepSeek-R1-Distill-Qwen-7B using LoRA on PeerRT, a new dataset of high-quality top AI conference reviews enriched with reasoning traces. We then apply Group Relative Policy Optimization (GRPO) to train two models: REMOR-H (with the human-aligned reward) and REMOR-U (with a uniform reward). Interestingly, the human-aligned reward penalizes aspects typically associated with strong reviews, leading REMOR-U to produce qualitatively more substantive feedback. Our results show that REMOR-U and REMOR-H achieve more than twice the average rewards of human reviews, non-reasoning state-of-the-art agentic multi-modal AI review systems, and general commercial LLM baselines. We found that while the best AI and human reviews are comparable in quality, REMOR avoids the long tail of low-quality human reviews. We discuss how reasoning is key to achieving these improvements and release the Human-aligned Peer Review Reward (HPRR) function, the Peer Review Reasoning-enriched Traces (PeerRT) dataset, and the REMOR models, which we believe can help spur progress in the area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11715v1">ConflictLens: LLM-Based Conflict Resolution Training in Romantic Relationship</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Romantic conflicts are often rooted in deep psychological factors such as coping styles, emotional responses, and communication habits. Existing systems tend to address surface-level behaviors or isolated events, offering limited support for understanding the underlying dynamics. We present ConflictLens, an interactive system that leverages psychological theory and large language models (LLMs) to help individuals analyze and reflect on the deeper mechanisms behind their conflicts. The system provides multi-level strategy recommendations and guided dialogue exercises, including annotation, rewriting, and continuation tasks. A case study demonstrates how ConflictLens supports emotional insight, improves relational understanding, and fosters more constructive communication. This work offers a novel approach to supporting self-awareness and growth in romantic relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00234v3">Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Improving Large Language Model (LLM) agents for sequential decision-making tasks typically requires extensive task-specific knowledge engineering--custom prompts, curated examples, and specialized observation/action spaces. We investigate a different approach where agents automatically improve by learning from their own successful experiences without human intervention. Our method constructs and refines a database of self-generated trajectories that serve as in-context examples for future tasks. Even naive accumulation of successful trajectories yields substantial performance gains across three diverse benchmarks: ALFWorld (73% to 89%), Wordcraft (55% to 64%), and InterCode-SQL (75% to 79%). These improvements exceed those achieved by upgrading from gpt-4o-mini to gpt-4o and match the performance of allowing multiple attempts per task. We further enhance this approach with two innovations: database-level curation using population-based training to propagate high-performing example collections, and exemplar-level curation that selectively retains trajectories based on their empirical utility as in-context examples. With these enhancements, our method achieves 93% success on ALFWorld--surpassing approaches that use more powerful LLMs and hand-crafted components. Our trajectory bootstrapping technique demonstrates that agents can autonomously improve through experience, offering a scalable alternative to labor-intensive knowledge engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11701v1">DMN-Guided Prompting: A Low-Code Framework for Controlling LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Large Language Models, Decision Model and Notation, Prompt Engineering, Automated Feedback
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown considerable potential in automating decision logic within knowledge-intensive processes. However, their effectiveness largely depends on the strategy and quality of prompting. Since decision logic is typically embedded in prompts, it becomes challenging for end users to modify or refine it. Decision Model and Notation (DMN) offers a standardized graphical approach for defining decision logic in a structured, user-friendly manner. This paper introduces a DMN-guided prompting framework that breaks down complex decision logic into smaller, manageable components, guiding LLMs through structured decision pathways. We implemented the framework in a graduate-level course where students submitted assignments. The assignments and DMN models representing feedback instructions served as inputs to our framework. The instructor evaluated the generated feedback and labeled it for performance assessment. Our approach demonstrated promising results, outperforming chain-of-thought (CoT) prompting. Students also responded positively to the generated feedback, reporting high levels of perceived usefulness in a survey based on the Technology Acceptance Model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11618v1">Benchmarking Spatiotemporal Reasoning in LLMs and Reasoning Models: Capabilities and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Spatiotemporal reasoning plays a key role in Cyber-Physical Systems (CPS). Despite advances in Large Language Models (LLMs) and Large Reasoning Models (LRMs), their capacity to reason about complex spatiotemporal signals remains underexplored. This paper proposes a hierarchical SpatioTemporal reAsoning benchmaRK, STARK, to systematically evaluate LLMs across three levels of reasoning complexity: state estimation (e.g., predicting field variables, localizing and tracking events in space and time), spatiotemporal reasoning over states (e.g., inferring spatial-temporal relationships), and world-knowledge-aware reasoning that integrates contextual and domain knowledge (e.g., intent prediction, landmark-aware navigation). We curate 26 distinct spatiotemporal tasks with diverse sensor modalities, comprising 14,552 challenges where models answer directly or by Python Code Interpreter. Evaluating 3 LRMs and 8 LLMs, we find LLMs achieve limited success in tasks requiring geometric reasoning (e.g., multilateration or triangulation), particularly as complexity increases. Surprisingly, LRMs show robust performance across tasks with various levels of difficulty, often competing or surpassing traditional first-principle-based methods. Our results show that in reasoning tasks requiring world knowledge, the performance gap between LLMs and LRMs narrows, with some LLMs even surpassing LRMs. However, the LRM o3 model continues to achieve leading performance across all evaluated tasks, a result attributed primarily to the larger size of the reasoning models. STARK motivates future innovations in model architectures and reasoning paradigms for intelligent CPS by providing a structured framework to identify limitations in the spatiotemporal reasoning of LLMs and LRMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11584v1">LLM Agents Are Hypersensitive to Nudges</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 33 pages, 28 figures
    </div>
    <details class="paper-abstract">
      LLMs are being set loose in complex, real-world environments involving sequential decision-making and tool use. Often, this involves making choices on behalf of human users. However, not much is known about the distribution of such choices, and how susceptible they are to different choice architectures. We perform a case study with a few such LLM models on a multi-attribute tabular decision-making problem, under canonical nudges such as the default option, suggestions, and information highlighting, as well as additional prompting strategies. We show that, despite superficial similarities to human choice distributions, such models differ in subtle but important ways. First, they show much higher susceptibility to the nudges. Second, they diverge in points earned, being affected by factors like the idiosyncrasy of available prizes. Third, they diverge in information acquisition strategies: e.g. incurring substantial cost to reveal too much information, or selecting without revealing any. Moreover, we show that simple prompt strategies like zero-shot chain of thought (CoT) can shift the choice distribution, and few-shot prompting with human data can induce greater alignment. Yet, none of these methods resolve the sensitivity of these models to nudges. Finally, we show how optimal nudges optimized with a human resource-rational model can similarly increase LLM performance for some models. All these findings suggest that behavioral tests are needed before deploying models as agents or assistants acting on behalf of users in complex environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11565v1">ACSE-Eval: Can LLMs threat model real-world cloud infrastructure?</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Submitted to the 39th Annual Conference on Neural Information Processing Systems
    </div>
    <details class="paper-abstract">
      While Large Language Models have shown promise in cybersecurity applications, their effectiveness in identifying security threats within cloud deployments remains unexplored. This paper introduces AWS Cloud Security Engineering Eval, a novel dataset for evaluating LLMs cloud security threat modeling capabilities. ACSE-Eval contains 100 production grade AWS deployment scenarios, each featuring detailed architectural specifications, Infrastructure as Code implementations, documented security vulnerabilities, and associated threat modeling parameters. Our dataset enables systemic assessment of LLMs abilities to identify security risks, analyze attack vectors, and propose mitigation strategies in cloud environments. Our evaluations on ACSE-Eval demonstrate that GPT 4.1 and Gemini 2.5 Pro excel at threat identification, with Gemini 2.5 Pro performing optimally in 0-shot scenarios and GPT 4.1 showing superior results in few-shot settings. While GPT 4.1 maintains a slight overall performance advantage, Claude 3.7 Sonnet generates the most semantically sophisticated threat models but struggles with threat categorization and generalization. To promote reproducibility and advance research in automated cybersecurity threat analysis, we open-source our dataset, evaluation metrics, and methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12636v3">MORepair: Teaching LLMs to Repair Code via Multi-Objective Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Within the realm of software engineering, specialized tasks on code, such as program repair, present unique challenges, necessitating fine-tuning Large language models~(LLMs) to unlock state-of-the-art performance. Fine-tuning approaches proposed in the literature for LLMs on program repair tasks generally overlook the need to reason about the logic behind code changes, beyond syntactic patterns in the data. High-performing fine-tuning experiments also usually come at very high computational costs. With MORepair, we propose a novel perspective on the learning focus of LLM fine-tuning for program repair: we not only adapt the LLM parameters to the syntactic nuances of the task of code transformation (objective 1), but we also specifically fine-tune the LLM with respect to the logical reason behind the code change in the training data (objective 2). Such a multi-objective fine-tuning will instruct LLMs to generate high-quality patches. We apply MORepair to fine-tune four open-source LLMs with different sizes and architectures. Experimental results on function-level and repository-level repair benchmarks show that the implemented fine-tuning effectively boosts LLM repair performance by 11.4% to 56.0%. We further show that our fine-tuning strategy yields superior performance compared to the state-of-the-art approaches, including standard fine-tuning, Fine-tune-CoT, and RepairLLaMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10066v1">Dark LLMs: The Growing Threat of Unaligned AI Models</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) rapidly reshape modern life, advancing fields from healthcare to education and beyond. However, alongside their remarkable capabilities lies a significant threat: the susceptibility of these models to jailbreaking. The fundamental vulnerability of LLMs to jailbreak attacks stems from the very data they learn from. As long as this training data includes unfiltered, problematic, or 'dark' content, the models can inherently learn undesirable patterns or weaknesses that allow users to circumvent their intended safety controls. Our research identifies the growing threat posed by dark LLMs models deliberately designed without ethical guardrails or modified through jailbreak techniques. In our research, we uncovered a universal jailbreak attack that effectively compromises multiple state-of-the-art models, enabling them to answer almost any question and produce harmful outputs upon request. The main idea of our attack was published online over seven months ago. However, many of the tested LLMs were still vulnerable to this attack. Despite our responsible disclosure efforts, responses from major LLM providers were often inadequate, highlighting a concerning gap in industry practices regarding AI safety. As model training becomes more accessible and cheaper, and as open-source LLMs proliferate, the risk of widespread misuse escalates. Without decisive intervention, LLMs may continue democratizing access to dangerous knowledge, posing greater risks than anticipated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09427v2">SafePath: Conformal Prediction for Safe LLM-Based Autonomous Navigation</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show growing promise in autonomous driving by reasoning over complex traffic scenarios to generate path plans. However, their tendencies toward overconfidence, and hallucinations raise critical safety concerns. We introduce SafePath, a modular framework that augments LLM-based path planning with formal safety guarantees using conformal prediction. SafePath operates in three stages. In the first stage, we use an LLM that generates a set of diverse candidate paths, exploring possible trajectories based on agent behaviors and environmental cues. In the second stage, SafePath filters out high-risk trajectories while guaranteeing that at least one safe option is included with a user-defined probability, through a multiple-choice question-answering formulation that integrates conformal prediction. In the final stage, our approach selects the path with the lowest expected collision risk when uncertainty is low or delegates control to a human when uncertainty is high. We theoretically prove that SafePath guarantees a safe trajectory with a user-defined probability, and we show how its human delegation rate can be tuned to balance autonomy and safety. Extensive experiments on nuScenes and Highway-env show that SafePath reduces planning uncertainty by 77\% and collision rates by up to 70\%, demonstrating effectiveness in making LLM-driven path planning more safer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19159v3">A Sliding Layer Merging Method for Efficient Depth-Wise Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Compared to width-wise pruning, depth-wise pruning can significantly accelerate inference in resource-constrained scenarios. However, treating the entire Transformer layer as the minimum pruning unit may degrade model performance by indiscriminately discarding the entire information of the layer. This paper reveals the ``Patch-like'' feature relationship between layers in large language models by analyzing the correlation of the outputs of different layers in the reproducing kernel Hilbert space. Building on this observation, we propose a sliding layer merging method that dynamically selects and fuses consecutive layers from top to bottom according to a pre-defined similarity threshold, thereby simplifying the model structure while maintaining its performance. Extensive experiments on LLMs with various architectures and different parameter scales show that our method outperforms existing pruning techniques in both zero-shot inference performance and retraining recovery quality after pruning. In particular, in the experiment with 35% pruning on the Vicuna-7B model, our method achieved a 1.654% improvement in average performance on zero-shot tasks compared to the existing method. Moreover, we further reveal the potential of combining depth pruning with width pruning to enhance the pruning effect. Our codes are available at https://github.com/920927/SLM-a-sliding-layer-merging-method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10013v1">DIF: A Framework for Benchmarking and Verifying Implicit Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 7 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) have risen in prominence over the past few years, there has been concern over the potential biases in LLMs inherited from the training data. Previous studies have examined how LLMs exhibit implicit bias, such as when response generation changes when different social contexts are introduced. We argue that this implicit bias is not only an ethical, but also a technical issue, as it reveals an inability of LLMs to accommodate extraneous information. However, unlike other measures of LLM intelligence, there are no standard methods to benchmark this specific subset of LLM bias. To bridge this gap, we developed a method for calculating an easily interpretable benchmark, DIF (Demographic Implicit Fairness), by evaluating preexisting LLM logic and math problem datasets with sociodemographic personas. We demonstrate that this method can statistically validate the presence of implicit bias in LLM behavior and find an inverse trend between question answering accuracy and implicit bias, supporting our argument.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10008v1">SVA-ICL: Improving LLM-based Software Vulnerability Assessment via In-Context Learning and Information Fusion</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Context: Software vulnerability assessment (SVA) is critical for identifying, evaluating, and prioritizing security weaknesses in software applications. Objective: Despite the increasing application of large language models (LLMs) in various software engineering tasks, their effectiveness in SVA remains underexplored. Method: To address this gap, we introduce a novel approach SVA-ICL, which leverages in-context learning (ICL) to enhance LLM performance. Our approach involves the selection of high-quality demonstrations for ICL through information fusion, incorporating both source code and vulnerability descriptions. For source code, we consider semantic, lexical, and syntactic similarities, while for vulnerability descriptions, we focus on textual similarity. Based on the selected demonstrations, we construct context prompts and consider DeepSeek-V2 as the LLM for SVA-ICL. Results: We evaluate the effectiveness of SVA-ICL using a large-scale dataset comprising 12,071 C/C++ vulnerabilities. Experimental results demonstrate that SVA-ICL outperforms state-of-the-art SVA baselines in terms of Accuracy, F1-score, and MCC measures. Furthermore, ablation studies highlight the significance of component customization in SVA-ICL, such as the number of demonstrations, the demonstration ordering strategy, and the optimal fusion ratio of different modalities. Conclusion: Our findings suggest that leveraging ICL with information fusion can effectively improve the effectiveness of LLM-based SVA, warranting further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09974v1">Analysing Safety Risks in LLMs Fine-Tuned with Pseudo-Malicious Cyber Security Data</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. We present a systematic evaluation of safety risks in fine-tuned LLMs for cyber security applications. Using the OWASP Top 10 for LLM Applications framework, we assessed seven open-source LLMs: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. Our evaluation shows that fine-tuning reduces safety resilience across all tested LLMs (e.g., the safety score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). We propose and evaluate a safety alignment approach that carefully rewords instruction-response pairs to include explicit safety precautions and ethical considerations. This approach demonstrates that it is possible to maintain or even improve model safety while preserving technical utility, offering a practical path forward for developing safer fine-tuning methodologies. This work offers a systematic evaluation for safety risks in LLMs, enabling safer adoption of generative AI in sensitive domains, and contributing towards the development of secure, trustworthy, and ethically aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09970v1">Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      The ReAct (Reasoning + Action) capability in large language models (LLMs) has become the foundation of modern agentic systems. Recent LLMs, such as DeepSeek-R1 and OpenAI o1/o3, exemplify this by emphasizing reasoning through the generation of ample intermediate tokens, which help build a strong premise before producing the final output tokens. In this paper, we introduce Pre-Act, a novel approach that enhances the agent's performance by creating a multi-step execution plan along with the detailed reasoning for the given user input. This plan incrementally incorporates previous steps and tool outputs, refining itself after each step execution until the final response is obtained. Our approach is applicable to both conversational and non-conversational agents. To measure the performance of task-oriented agents comprehensively, we propose a two-level evaluation framework: (1) turn level and (2) end-to-end. Our turn-level evaluation, averaged across five models, shows that our approach, Pre-Act, outperforms ReAct by 70% in Action Recall on the Almita dataset. While this approach is effective for larger models, smaller models crucial for practical applications, where latency and cost are key constraints, often struggle with complex reasoning tasks required for agentic systems. To address this limitation, we fine-tune relatively small models such as Llama 3.1 (8B & 70B) using the proposed Pre-Act approach. Our experiments show that the fine-tuned 70B model outperforms GPT-4, achieving a 69.5% improvement in action accuracy (turn-level) and a 28% improvement in goal completion rate (end-to-end) on the Almita (out-of-domain) dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07440v2">Model Utility Law: Evaluating LLMs beyond Performance through Mechanism Interpretable Metric</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become indispensable across academia, industry, and daily applications, yet current evaluation methods struggle to keep pace with their rapid development. One core challenge of evaluation in the large language model (LLM) era is the generalization issue: how to infer a model's near-unbounded abilities from inevitably bounded benchmarks. We address this challenge by proposing Model Utilization Index (MUI), a mechanism interpretability enhanced metric that complements traditional performance scores. MUI quantifies the effort a model expends on a task, defined as the proportion of activated neurons or features during inference. Intuitively, a truly capable model should achieve higher performance with lower effort. Extensive experiments across popular LLMs reveal a consistent inverse logarithmic relationship between MUI and performance, which we formulate as the Utility Law. From this law we derive four practical corollaries that (i) guide training diagnostics, (ii) expose data contamination issue, (iii) enable fairer model comparisons, and (iv) design model-specific dataset diversity. Our code can be found at https://github.com/ALEX-nlp/MUI-Eva.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09921v1">PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at \href{https://github.com/redwyd/PrivacyJailbreak}{https://github.com/redwyd/PrivacyJailbreak}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01698v3">Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 19 Pages, https://github.com/abhibambhaniya/GenZ-LLM-Analyzer, https://genz-llm-analyzer.streamlit.app/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance across a wide range of applications, often outperforming human experts. However, deploying these gigantic models efficiently for diverse inference use cases requires carefully designed hardware platforms with ample computing, memory, and network resources. With constant innovation in LLM serving optimizations and model architecture evolving at breakneck speed, the hardware requirements to meet Service Level Objectives (SLOs) remain an open research question. To answer the question, we present an analytical tool, GenZ, to efficiently navigate the relationship between diverse LLM model architectures(Dense, GQA, MoE, Mamba), LLM serving optimizations(Chunking, Speculative decoding, quanitization), and AI platform design parameters. Our tool estimates LLM inference performance metrics for the given scenario. We have validated against real hardware platforms running various different LLM models, achieving a max geomean error of 5.82.We use GenZ to identify compute, memory capacity, memory bandwidth, network latency, and network bandwidth requirements across diverse LLM inference use cases. We also study diverse architectural choices in use today (inspired by LLM serving platforms from several vendors) to help inform computer architects designing next-generation AI hardware accelerators and platforms. The trends and insights derived from GenZ can guide AI engineers deploying LLMs as well as computer architects designing next-generation hardware accelerators and platforms. Ultimately, this work sheds light on the platform design considerations for unlocking the full potential of large language models across a spectrum of applications. The source code is available at https://github.com/abhibambhaniya/GenZ-LLM-Analyzer . Users can also be tried it on at https://genz-llm-analyzer.streamlit.app/ without any setup on your web browser.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09901v1">Comparing Exploration-Exploitation Strategies of LLMs and Humans: Insights from Standard Multi-armed Bandit Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate or automate human behavior in complex sequential decision-making tasks. A natural question is then whether LLMs exhibit similar decision-making behavior to humans, and can achieve comparable (or superior) performance. In this work, we focus on the exploration-exploitation (E&E) tradeoff, a fundamental aspect of dynamic decision-making under uncertainty. We employ canonical multi-armed bandit (MAB) tasks introduced in the cognitive science and psychiatry literature to conduct a comparative study of the E&E strategies of LLMs, humans, and MAB algorithms. We use interpretable choice models to capture the E&E strategies of the agents and investigate how explicit reasoning, through both prompting strategies and reasoning-enhanced models, shapes LLM decision-making. We find that reasoning shifts LLMs toward more human-like behavior, characterized by a mix of random and directed exploration. In simple stationary tasks, reasoning-enabled LLMs exhibit similar levels of random and directed exploration compared to humans. However, in more complex, non-stationary environments, LLMs struggle to match human adaptability, particularly in effective directed exploration, despite achieving similar regret in certain scenarios. Our findings highlight both the promise and limits of LLMs as simulators of human behavior and tools for automated decision-making and point to potential areas of improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.06018v4">TARGET: Automated Scenario Generation from Traffic Rules for Testing Autonomous Vehicles via Validated LLM-Guided Knowledge Extraction</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Recent incidents with autonomous vehicles highlight the need for rigorous testing to ensure safety and robustness. Constructing test scenarios for autonomous driving systems (ADSs), however, is labor-intensive. We propose TARGET, an end-to-end framework that automatically generates test scenarios from traffic rules. To address complexity, we leverage a Large Language Model (LLM) to extract knowledge from traffic rules. To mitigate hallucinations caused by large context during input processing, we introduce a domain-specific language (DSL) designed to be syntactically simple and compositional. This design allows the LLM to learn and generate test scenarios in a modular manner while enabling syntactic and semantic validation for each component. Based on these validated representations, TARGET synthesizes executable scripts to render scenarios in simulation. Evaluated seven ADSs with 284 scenarios derived from 54 traffic rules, TARGET uncovered 610 rule violations, collisions, and other issues. For each violation, TARGET generates scenario recordings and detailed logs, aiding root cause analysis. Two identified issues were confirmed by ADS developers: one linked to an existing bug report and the other to limited ADS functionality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09986v2">From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Motivated by the remarkable success of artificial intelligence (AI) across diverse fields, the application of AI to solve scientific problems, often formulated as partial differential equations (PDEs), has garnered increasing attention. While most existing research concentrates on theoretical properties (such as well-posedness, regularity, and continuity) of the solutions, alongside direct AI-driven methods for solving PDEs, the challenge of uncovering symbolic relationships within these equations remains largely unexplored. In this paper, we propose leveraging large language models (LLMs) to learn such symbolic relationships. Our results demonstrate that LLMs can effectively predict the operators involved in PDE solutions by utilizing the symbolic information in the PDEs both theoretically and numerically. Furthermore, we show that discovering these symbolic relationships can substantially improve both the efficiency and accuracy of symbolic machine learning for finding analytical approximation of PDE solutions, delivering a fully interpretable solution pipeline. This work opens new avenues for understanding the symbolic structure of scientific problems and advancing their solution processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14609v2">DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 11 pages, 6 figures. SIGIR '25 Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval July 13--18, 2025 Padua, Italy
    </div>
    <details class="paper-abstract">
      Conversational Search (CS) involves retrieving relevant documents from a corpus while considering the conversational context, integrating retrieval with context modeling. Recent advancements in Large Language Models (LLMs) have significantly enhanced CS by enabling query rewriting based on conversational context. However, employing LLMs during inference poses efficiency challenges. Existing solutions mitigate this issue by distilling embeddings derived from human-rewritten queries, focusing primarily on learning the context modeling task. These methods, however, often separate the contrastive retrieval task from the distillation process, treating it as an independent loss term. To overcome these limitations, we introduce DiSCo (Distillation of Sparse Conversational retrieval), a novel approach that unifies retrieval and context modeling through a relaxed distillation objective. Instead of relying exclusively on representation learning, our method distills similarity scores between conversations and documents, providing more freedom in the representation space and better leveraging the contrastive nature of document relevance. Extensive experiments on Learned Sparse Retrieval (LSR) across five CS datasets demonstrate that DiSCo achieves substantial improvements in both in-domain and out-of-domain retrieval tasks, achieving up to a six-point gain in recall for out-of-domain datasets over state-of-the-art methods. Additionally, DiSCo employs a multi-teacher distillation strategy, using multiple LLMs as teachers, further enhancing performance and surpassing the individual teachers in in-domain settings. Furthermore, analysis of model sparsity reveals that DiSCo allows for more effective control over the sparsity of the trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10708v1">SafeTrans: LLM-assisted Transpilation from C to Rust</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Rust is a strong contender for a memory-safe alternative to C as a "systems" programming language, but porting the vast amount of existing C code to Rust is a daunting task. In this paper, we evaluate the potential of large language models (LLMs) to automate the transpilation of C code to idiomatic Rust, while ensuring that the generated code mitigates any memory-related vulnerabilities present in the original code. To that end, we present the design and implementation of SafeTrans, a framework that uses LLMs to i) transpile C code into Rust and ii) iteratively fix any compilation and runtime errors in the resulting code. A key novelty of our approach is the introduction of a few-shot guided repair technique for translation errors, which provides contextual information and example code snippets for specific error types, guiding the LLM toward the correct solution. Another novel aspect of our work is the evaluation of the security implications of the transpilation process, i.e., whether potential vulnerabilities in the original C code have been properly addressed in the translated Rust code. We experimentally evaluated SafeTrans with six leading LLMs and a set of 2,653 C programs accompanied by comprehensive unit tests, which were used for validating the correctness of the translated code. Our results show that our iterative repair strategy improves the rate of successful translations from 54% to 80% for the best-performing LLM (GPT-4o), and that all types of identified vulnerabilities in the original C code are effectively mitigated in the translated Rust code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09598v2">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      This paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.43 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: Although AI is becoming cheaper and faster, its global adoption drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10681v1">Towards an LLM-powered Social Digital Twinning Platform</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 13 pages, 3 figures, 23rd International Conference on Practical applications of Agents and Multi-Agent Systems (PAAMS 2025)
    </div>
    <details class="paper-abstract">
      We present Social Digital Twinner, an innovative social simulation tool for exploring plausible effects of what-if scenarios in complex adaptive social systems. The architecture is composed of three seamlessly integrated parts: a data infrastructure featuring real-world data and a multi-dimensionally representative synthetic population of citizens, an LLM-enabled agent-based simulation engine, and a user interface that enable intuitive, natural language interactions with the simulation engine and the artificial agents (i.e. citizens). Social Digital Twinner facilitates real-time engagement and empowers stakeholders to collaboratively design, test, and refine intervention measures. The approach is promoting a data-driven and evidence-based approach to societal problem-solving. We demonstrate the tool's interactive capabilities by addressing the critical issue of youth school dropouts in Kragero, Norway, showcasing its ability to create and execute a dedicated social digital twin using natural language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10670v1">Interpretable Risk Mitigation in LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Autonomous agents powered by large language models (LLMs) enable novel use cases in domains where responsible action is increasingly important. Yet the inherent unpredictability of LLMs raises safety concerns about agent reliability. In this work, we explore agent behaviour in a toy, game-theoretic environment based on a variation of the Iterated Prisoner's Dilemma. We introduce a strategy-modification method-independent of both the game and the prompt-by steering the residual stream with interpretable features extracted from a sparse autoencoder latent space. Steering with the good-faith negotiation feature lowers the average defection probability by 28 percentage points. We also identify feasible steering ranges for several open-source LLM agents. Finally, we hypothesise that game-theoretic evaluation of LLM agents, combined with representation-steering alignment, can generalise to real-world applications on end-user devices and embodied platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10597v1">Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Reward models (RMs) are essential for aligning large language models (LLMs) with human values. However, noisy preferences in human feedback often lead to reward misgeneralization, where RMs overfit to spurious patterns and provide misleading signals during policy optimization. We systematically analyze the training dynamics of preference pairs and identify that noisy examples are harder to fit and introduce instability. Empirical evidence shows that LLMs optimized using reward models trained on full noisy datasets perform worse than those trained on filtered, high-quality preferences. To address this, we propose Collaborative Reward Modeling (CRM), an online framework that enhances robustness by combining peer review and curriculum learning. Two reward models are trained in parallel and assess each other's data selections to filter out potential noise. Curriculum learning structures the preference data from easy to hard, ensuring synchronized training and stable feedback. Extensive experiments demonstrate that CRM improves generalization, with up to 9.94 points of accuracy gain on RewardBench under 40 percent label noise. CRM is also compatible with implicit-reward alignment methods, offering a practical and versatile strategy for robust alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10593v1">LLM-Explorer: Towards Efficient and Affordable LLM-based Exploration for Mobile Apps</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Accepted by MobiCom 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have opened new opportunities for automated mobile app exploration, an important and challenging problem that used to suffer from the difficulty of generating meaningful UI interactions. However, existing LLM-based exploration approaches rely heavily on LLMs to generate actions in almost every step, leading to a huge cost of token fees and computational resources. We argue that such extensive usage of LLMs is neither necessary nor effective, since many actions during exploration do not require, or may even be biased by the abilities of LLMs. Further, based on the insight that a precise and compact knowledge plays the central role for effective exploration, we introduce LLM-Explorer, a new exploration agent designed for efficiency and affordability. LLM-Explorer uses LLMs primarily for maintaining the knowledge instead of generating actions, and knowledge is used to guide action generation in a LLM-less manner. Based on a comparison with 5 strong baselines on 20 typical apps, LLM-Explorer was able to achieve the fastest and highest coverage among all automated app explorers, with over 148x lower cost than the state-of-the-art LLM-based approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10495v1">RouteNator: A Router-Based Multi-Modal Architecture for Generating Synthetic Training Data for Function Calling LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing
    </div>
    <details class="paper-abstract">
      This paper addresses fine-tuning Large Language Models (LLMs) for function calling tasks when real user interaction data is unavailable. In digital content creation tools, where users express their needs through natural language queries that must be mapped to API calls, the lack of real-world task-specific data and privacy constraints for training on it necessitate synthetic data generation. Existing approaches to synthetic data generation fall short in diversity and complexity, failing to replicate real-world data distributions and leading to suboptimal performance after LLM fine-tuning. We present a novel router-based architecture that leverages domain resources like content metadata and structured knowledge graphs, along with text-to-text and vision-to-text language models to generate high-quality synthetic training data. Our architecture's flexible routing mechanism enables synthetic data generation that matches observed real-world distributions, addressing a fundamental limitation of traditional approaches. Evaluation on a comprehensive set of real user queries demonstrates significant improvements in both function classification accuracy and API parameter selection. Models fine-tuned with our synthetic data consistently outperform traditional approaches, establishing new benchmarks for function calling tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10490v1">Campus AI vs Commercial AI: A Late-Breaking Study on How LLM As-A-Service Customizations Shape Trust and Usage Patterns</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      As the use of Large Language Models (LLMs) by students, lecturers and researchers becomes more prevalent, universities - like other organizations - are pressed to develop coherent AI strategies. LLMs as-a-Service (LLMaaS) offer accessible pre-trained models, customizable to specific (business) needs. While most studies prioritize data, model, or infrastructure adaptations (e.g., model fine-tuning), we focus on user-salient customizations, like interface changes and corporate branding, which we argue influence users' trust and usage patterns. This study serves as a functional prequel to a large-scale field study in which we examine how students and employees at a German university perceive and use their institution's customized LLMaaS compared to ChatGPT. The goals of this prequel are to stimulate discussions on psychological effects of LLMaaS customizations and refine our research approach through feedback. Our forthcoming findings will deepen the understanding of trust dynamics in LLMs, providing practical guidance for organizations considering LLMaaS deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v4">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 9 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10425v1">Learning to Think: Information-Theoretic Reinforcement Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at complex tasks thanks to advances in reasoning abilities. However, existing methods overlook the trade-off between reasoning effectiveness and computational efficiency, often encouraging unnecessarily long reasoning chains and wasting tokens. To address this, we propose Learning to Think (L2T), an information-theoretic reinforcement fine-tuning framework for LLMs to make the models achieve optimal reasoning with fewer tokens. Specifically, L2T treats each query-response interaction as a hierarchical session of multiple episodes and proposes a universal dense process reward, i.e., quantifies the episode-wise information gain in parameters, requiring no extra annotations or task-specific evaluators. We propose a method to quickly estimate this reward based on PAC-Bayes bounds and the Fisher information matrix. Theoretical analyses show that it significantly reduces computational complexity with high estimation accuracy. By immediately rewarding each episode's contribution and penalizing excessive updates, L2T optimizes the model via reinforcement learning to maximize the use of each episode and achieve effective updates. Empirical results on various reasoning benchmarks and base models demonstrate the advantage of L2T across different tasks, boosting both reasoning effectiveness and efficiency.
    </details>
</div>
