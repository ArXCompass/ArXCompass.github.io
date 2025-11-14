# llm - 2024_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02376v2">Retaining Key Information under High Compression Ratios: Query-Guided Compressor for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 Accepted to ACL 2024
    </div>
    <details class="paper-abstract">
      The growing popularity of Large Language Models has sparked interest in context compression for Large Language Models (LLMs). However, the performance of previous methods degrades dramatically as compression ratios increase, sometimes even falling to the closed-book level. This decline can be attributed to the loss of key information during the compression process. Our preliminary study supports this hypothesis, emphasizing the significance of retaining key information to maintain model performance under high compression ratios. As a result, we introduce Query-Guided Compressor (QGC), which leverages queries to guide the context compression process, effectively preserving key information within the compressed context. Additionally, we employ a dynamic compression strategy. We validate the effectiveness of our proposed QGC on the Question Answering task, including NaturalQuestions, TriviaQA, and HotpotQA datasets. Experimental results show that QGC can consistently perform well even at high compression ratios, which also offers significant benefits in terms of inference cost and throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13671v2">KInIT at SemEval-2024 Task 8: Fine-tuned LLMs for Multilingual Machine-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 SemEval-2024 Task 8
    </div>
    <details class="paper-abstract">
      SemEval-2024 Task 8 is focused on multigenerator, multidomain, and multilingual black-box machine-generated text detection. Such a detection is important for preventing a potential misuse of large language models (LLMs), the newest of which are very capable in generating multilingual human-like texts. We have coped with this task in multiple ways, utilizing language identification and parameter-efficient fine-tuning of smaller LLMs for text classification. We have further used the per-language classification-threshold calibration to uniquely combine fine-tuned models predictions with statistical detection metrics to improve generalization of the system detection performance. Our submitted method achieved competitive results, ranking at the fourth place, just under 1 percentage point behind the winner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11930v1">A Critical Study of What Code-LLMs (Do Not) Learn</a></div>
    <div class="paper-meta">
      📅 2024-06-17
    </div>
    <details class="paper-abstract">
      Large Language Models trained on code corpora (code-LLMs) have demonstrated impressive performance in various coding assistance tasks. However, despite their increased size and training dataset, code-LLMs still have limitations such as suggesting codes with syntactic errors, variable misuse etc. Some studies argue that code-LLMs perform well on coding tasks because they use self-attention and hidden representations to encode relations among input tokens. However, previous works have not studied what code properties are not encoded by code-LLMs. In this paper, we conduct a fine-grained analysis of attention maps and hidden representations of code-LLMs. Our study indicates that code-LLMs only encode relations among specific subsets of input tokens. Specifically, by categorizing input tokens into syntactic tokens and identifiers, we found that models encode relations among syntactic tokens and among identifiers, but they fail to encode relations between syntactic tokens and identifiers. We also found that fine-tuned models encode these relations poorly compared to their pre-trained counterparts. Additionally, larger models with billions of parameters encode significantly less information about code than models with only a few hundred million parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01964v3">Position: Understanding LLMs Requires More Than Statistical Generalization</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 Accepted as a position paper at ICML2024, Code: https://github.com/rpatrik96/llm-non-identifiability
    </div>
    <details class="paper-abstract">
      The last decade has seen blossoming research in deep learning theory attempting to answer, "Why does deep learning generalize?" A powerful shift in perspective precipitated this progress: the study of overparametrized models in the interpolation regime. In this paper, we argue that another perspective shift is due, since some of the desirable qualities of LLMs are not a consequence of good statistical generalization and require a separate theoretical explanation. Our core argument relies on the observation that AR probabilistic models are inherently non-identifiable: models zero or near-zero KL divergence apart -- thus, equivalent test loss -- can exhibit markedly different behaviors. We support our position with mathematical examples and empirical observations, illustrating why non-identifiability has practical relevance through three case studies: (1) the non-identifiability of zero-shot rule extrapolation; (2) the approximate non-identifiability of in-context learning; and (3) the non-identifiability of fine-tunability. We review promising research directions focusing on LLM-relevant generalization measures, transferability, and inductive biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16891v1">Cultural Value Differences of LLMs: Prompt, Language, and Model Size</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Our study aims to identify behavior patterns in cultural values exhibited by large language models (LLMs). The studied variants include question ordering, prompting language, and model size. Our experiments reveal that each tested LLM can efficiently behave with different cultural values. More interestingly: (i) LLMs exhibit relatively consistent cultural values when presented with prompts in a single language. (ii) The prompting language e.g., Chinese or English, can influence the expression of cultural values. The same question can elicit divergent cultural values when the same LLM is queried in a different language. (iii) Differences in sizes of the same model (e.g., Llama2-7B vs 13B vs 70B) have a more significant impact on their demonstrated cultural values than model differences (e.g., Llama2 vs Mixtral). Our experiments reveal that query language and model size of LLM are the main factors resulting in cultural value differences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11424v1">Evaluating the Efficacy of Open-Source LLMs in Enterprise-Specific RAG Systems: A Comparative Study of Performance and Scalability</a></div>
    <div class="paper-meta">
      📅 2024-06-17
    </div>
    <details class="paper-abstract">
      This paper presents an analysis of open-source large language models (LLMs) and their application in Retrieval-Augmented Generation (RAG) tasks, specific for enterprise-specific data sets scraped from their websites. With the increasing reliance on LLMs in natural language processing, it is crucial to evaluate their performance, accessibility, and integration within specific organizational contexts. This study examines various open-source LLMs, explores their integration into RAG frameworks using enterprise-specific data, and assesses the performance of different open-source embeddings in enhancing the retrieval and generation process. Our findings indicate that open-source LLMs, combined with effective embedding techniques, can significantly improve the accuracy and efficiency of RAG systems, offering a viable alternative to proprietary solutions for enterprises.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10811v2">Quantifying the Persona Effect in LLM Simulations</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 ACL 2024 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable promise in simulating human language and behavior. This study investigates how integrating persona variables-demographic, social, and behavioral factors-impacts LLMs' ability to simulate diverse perspectives. We find that persona variables account for <10% variance in annotations in existing subjective NLP datasets. Nonetheless, incorporating persona variables via prompting in LLMs provides modest but statistically significant improvements. Persona prompting is most effective in samples where many annotators disagree, but their disagreements are relatively minor. Notably, we find a linear relationship in our setting: the stronger the correlation between persona variables and human annotations, the more accurate the LLM predictions are using persona prompting. In a zero-shot setting, a powerful 70b model with persona prompting captures 81% of the annotation variance achievable by linear regression trained on ground truth annotations. However, for most subjective NLP datasets, where persona variables have limited explanatory power, the benefits of persona prompting are limited.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06211v3">A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 This is the long version of the corresponding survey paper accepted by KDD2024
    </div>
    <details class="paper-abstract">
      As one of the most advanced techniques in AI, Retrieval-Augmented Generation (RAG) can offer reliable and up-to-date external knowledge, providing huge convenience for numerous tasks. Particularly in the era of AI-Generated Content (AIGC), the powerful capacity of retrieval in providing additional knowledge enables RAG to assist existing generative AI in producing high-quality outputs. Recently, Large Language Models (LLMs) have demonstrated revolutionary abilities in language understanding and generation, while still facing inherent limitations, such as hallucinations and out-of-date internal knowledge. Given the powerful abilities of RAG in providing the latest and helpful auxiliary information, Retrieval-Augmented Large Language Models (RA-LLMs) have emerged to harness external and authoritative knowledge bases, rather than solely relying on the model's internal knowledge, to augment the generation quality of LLMs. In this survey, we comprehensively review existing research studies in RA-LLMs, covering three primary technical perspectives: architectures, training strategies, and applications. As the preliminary knowledge, we briefly introduce the foundations and recent advances of LLMs. Then, to illustrate the practical significance of RAG for LLMs, we systematically review mainstream relevant work by their architectures, training strategies, and application areas, detailing specifically the challenges of each and the corresponding capabilities of RA-LLMs. Finally, to deliver deeper insights, we discuss current limitations and several promising directions for future research. Updated information about this survey can be found at https://advanced-recommender-systems.github.io/RAG-Meets-LLMs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11257v1">ExCP: Extreme LLM Checkpoint Compression via Weight-Momentum Joint Shrinking</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 ICML 2024 oral
    </div>
    <details class="paper-abstract">
      Large language models (LLM) have recently attracted significant attention in the field of artificial intelligence. However, the training process of these models poses significant challenges in terms of computational and storage capacities, thus compressing checkpoints has become an urgent problem. In this paper, we propose a novel Extreme Checkpoint Compression (ExCP) framework, which significantly reduces the required storage of training checkpoints while achieving nearly lossless performance. We first calculate the residuals of adjacent checkpoints to obtain the essential but sparse information for higher compression ratio. To further excavate the redundancy parameters in checkpoints, we then propose a weight-momentum joint shrinking method to utilize another important information during the model optimization, i.e., momentum. In particular, we exploit the information of both model and optimizer to discard as many parameters as possible while preserving critical information to ensure optimal performance. Furthermore, we utilize non-uniform quantization to further compress the storage of checkpoints. We extensively evaluate our proposed ExCP framework on several models ranging from 410M to 7B parameters and demonstrate significant storage reduction while maintaining strong performance. For instance, we achieve approximately $70\times$ compression for the Pythia-410M model, with the final performance being as accurate as the original model on various downstream tasks. Codes will be available at https://github.com/Gaffey/ExCP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14469v2">SnapKV: LLM Knows What You are Looking for Before Generation</a></div>
    <div class="paper-meta">
      📅 2024-06-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made remarkable progress in processing extensive contexts, with the Key-Value (KV) cache playing a vital role in enhancing their performance. However, the growth of the KV cache in response to increasing input length poses challenges to memory and time efficiency. To address this problem, this paper introduces SnapKV, an innovative and fine-tuning-free approach that efficiently minimizes KV cache size while still delivering comparable performance in real-world applications. We discover that each attention head in the model consistently focuses on specific prompt attention features during generation. Meanwhile, this robust pattern can be obtained from an 'observation' window located at the end of the prompts. Drawing on this insight, SnapKV automatically compresses KV caches by selecting clustered important KV positions for each attention head. Our approach significantly reduces the growing computational overhead and memory footprint when processing long input sequences. Specifically, SnapKV achieves a consistent decoding speed with a 3.6x increase in generation speed and an 8.2x enhancement in memory efficiency compared to the baseline when processing inputs of 16K tokens. At the same time, it maintains comparable performance to the baseline models across 16 long sequence datasets. Moreover, SnapKV can process up to 380K context tokens on a single A100-80GB GPU using HuggingFace implementation with minor changes, exhibiting only a negligible accuracy drop in the Needle-in-a-Haystack test. Further comprehensive studies suggest SnapKV's potential for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.11657v1">Can LLM be a Personalized Judge?</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 Our code is available at https://github.com/dong-river/Personalized-Judge
    </div>
    <details class="paper-abstract">
      Ensuring that large language models (LLMs) reflect diverse user values and preferences is crucial as their user bases expand globally. It is therefore encouraging to see the growing interest in LLM personalization within the research community. However, current works often rely on the LLM-as-a-Judge approach for evaluation without thoroughly examining its validity. In this paper, we investigate the reliability of LLM-as-a-Personalized-Judge, asking LLMs to judge user preferences based on personas. Our findings suggest that directly applying LLM-as-a-Personalized-Judge is less reliable than previously assumed, showing low and inconsistent agreement with human ground truth. The personas typically used are often overly simplistic, resulting in low predictive power. To address these issues, we introduce verbal uncertainty estimation into the LLM-as-a-Personalized-Judge pipeline, allowing the model to express low confidence on uncertain judgments. This adjustment leads to much higher agreement (above 80%) on high-certainty samples for binary tasks. Through human evaluation, we find that the LLM-as-a-Personalized-Judge achieves comparable performance to third-party humans evaluation and even surpasses human performance on high-certainty samples. Our work indicates that certainty-enhanced LLM-as-a-Personalized-Judge offers a promising direction for developing more reliable and scalable methods for evaluating LLM personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11093v1">RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning based on Emotional Information</a></div>
    <div class="paper-meta">
      📅 2024-06-16
    </div>
    <details class="paper-abstract">
      Misinformation is prevalent in various fields such as education, politics, health, etc., causing significant harm to society. However, current methods for cross-domain misinformation detection rely on time and resources consuming fine-tuning and complex model structures. With the outstanding performance of LLMs, many studies have employed them for misinformation detection. Unfortunately, they focus on in-domain tasks and do not incorporate significant sentiment and emotion features (which we jointly call affect). In this paper, we propose RAEmoLLM, the first retrieval augmented (RAG) LLMs framework to address cross-domain misinformation detection using in-context learning based on affective information. It accomplishes this by applying an emotion-aware LLM to construct a retrieval database of affective embeddings. This database is used by our retrieval module to obtain source-domain samples, which are subsequently used for the inference module's in-context few-shot learning to detect target domain misinformation. We evaluate our framework on three misinformation benchmarks. Results show that RAEmoLLM achieves significant improvements compared to the zero-shot method on three datasets, with the highest increases of 20.69%, 23.94%, and 39.11% respectively. This work will be released on https://github.com/lzw108/RAEmoLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12934v1">Current state of LLM Risks and AI Guardrails</a></div>
    <div class="paper-meta">
      📅 2024-06-16
      | 💬 Independent study, Exploring LLMs, Deploying LLMs and their Risks
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly sophisticated, leading to widespread deployment in sensitive applications where safety and reliability are paramount. However, LLMs have inherent risks accompanying them, including bias, potential for unsafe actions, dataset poisoning, lack of explainability, hallucinations, and non-reproducibility. These risks necessitate the development of "guardrails" to align LLMs with desired behaviors and mitigate potential harm. This work explores the risks associated with deploying LLMs and evaluates current approaches to implementing guardrails and model alignment techniques. We examine intrinsic and extrinsic bias evaluation methods and discuss the importance of fairness metrics for responsible AI development. The safety and reliability of agentic LLMs (those capable of real-world actions) are explored, emphasizing the need for testability, fail-safes, and situational awareness. Technical strategies for securing LLMs are presented, including a layered protection model operating at external, secondary, and internal levels. System prompts, Retrieval-Augmented Generation (RAG) architectures, and techniques to minimize bias and protect privacy are highlighted. Effective guardrail design requires a deep understanding of the LLM's intended use case, relevant regulations, and ethical considerations. Striking a balance between competing requirements, such as accuracy and privacy, remains an ongoing challenge. This work underscores the importance of continuous research and development to ensure the safe and responsible use of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10340v4">Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics</a></div>
    <div class="paper-meta">
      📅 2024-06-16
    </div>
    <details class="paper-abstract">
      In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works focus on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation and navigation. Despite these improvements, analyzing the safety of such systems remains underexplored yet extremely critical. LLMs and VLMs are highly susceptible to adversarial inputs, prompting a significant inquiry into the safety of robotic systems. This concern is important because robotics operate in the physical world where erroneous actions can result in severe consequences. This paper explores this issue thoroughly, presenting a mathematical formulation of potential attacks on LLM/VLM-based robotic systems and offering experimental evidence of the safety challenges. Our empirical findings highlight a significant vulnerability: simple modifications to the input can drastically reduce system effectiveness. Specifically, our results demonstrate an average performance deterioration of 19.4% under minor input prompt modifications and a more alarming 29.1% under slight perceptual changes. These findings underscore the urgent need for robust countermeasures to ensure the safe and reliable deployment of advanced LLM/VLM-based robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11047v1">Enhancing Supermarket Robot Interaction: A Multi-Level LLM Conversational Interface for Handling Diverse Customer Intents</a></div>
    <div class="paper-meta">
      📅 2024-06-16
    </div>
    <details class="paper-abstract">
      This paper presents the design and evaluation of a novel multi-level LLM interface for supermarket robots to assist customers. The proposed interface allows customers to convey their needs through both generic and specific queries. While state-of-the-art systems like OpenAI's GPTs are highly adaptable and easy to build and deploy, they still face challenges such as increased response times and limitations in strategic control of the underlying model for tailored use-case and cost optimization. Driven by the goal of developing faster and more efficient conversational agents, this paper advocates for using multiple smaller, specialized LLMs fine-tuned to handle different user queries based on their specificity and user intent. We compare this approach to a specialized GPT model powered by GPT-4 Turbo, using the Artificial Social Agent Questionnaire (ASAQ) and qualitative participant feedback in a counterbalanced within-subjects experiment. Our findings show that our multi-LLM chatbot architecture outperformed the benchmarked GPT model across all 13 measured criteria, with statistically significant improvements in four key areas: performance, user satisfaction, user-agent partnership, and self-image enhancement. The paper also presents a method for supermarket robot navigation by mapping the final chatbot response to correct shelf numbers, enabling the robot to sequentially navigate towards the respective products, after which lower-level robot perception, control, and planning can be used for automated object retrieval. We hope this work encourages more efforts into using multiple, specialized smaller models instead of relying on a single powerful, but more expensive and slower model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00165v2">TELEClass: Taxonomy Enrichment and LLM-Enhanced Hierarchical Text Classification with Minimal Supervision</a></div>
    <div class="paper-meta">
      📅 2024-06-16
    </div>
    <details class="paper-abstract">
      Hierarchical text classification aims to categorize each document into a set of classes in a label taxonomy. Most earlier works focus on fully or semi-supervised methods that require a large amount of human annotated data which is costly and time-consuming to acquire. To alleviate human efforts, in this paper, we work on hierarchical text classification with the minimal amount of supervision: using the sole class name of each node as the only supervision. Recently, large language models (LLM) show competitive performance on various tasks through zero-shot prompting, but this method performs poorly in the hierarchical setting, because it is ineffective to include the large and structured label space in a prompt. On the other hand, previous weakly-supervised hierarchical text classification methods only utilize the raw taxonomy skeleton and ignore the rich information hidden in the text corpus that can serve as additional class-indicative features. To tackle the above challenges, we propose TELEClass, Taxonomy Enrichment and LLM-Enhanced weakly-supervised hierarchical text Classification, which (1) automatically enriches the label taxonomy with class-indicative terms to facilitate classifier training and (2) utilizes LLMs for both data annotation and creation tailored for the hierarchical label space. Experiments show that TELEClass can outperform previous weakly-supervised methods and LLM-based zero-shot prompting methods on two public datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.12961v2">Eloquent: A More Robust Transmission Scheme for LLM Token Streaming</a></div>
    <div class="paper-meta">
      📅 2024-06-16
      | 💬 In SIGCOMM Workshop on Networks for AI Computing (NAIC '24)
    </div>
    <details class="paper-abstract">
      To render each generated token in real-time for users, the Large Language Model (LLM) server generates tokens one by one and streams each token (or group of a few tokens) through the network to the user right after generation, which we refer to as LLM token streaming. However, under unstable network conditions, the LLM token streaming experience could suffer greatly from stalls since one packet loss could block the rendering of later tokens even if the packets containing them arrive on time. With a measurement study, we show that current applications suffer from increased stalls under unstable networks. For this emerging token streaming problem in LLM Chatbots that differs from previous multimedia and text applications, we propose a novel transmission scheme, called Eloquent, which puts newly generated tokens as well as currently unacknowledged tokens in the next outgoing packet. This ensures that each packet contains some new tokens and, in the meantime, is independently rendered when received, avoiding the aforementioned stalls caused by missing packets. Through simulation under various networks, we show Eloquent reduces stall ratio (proportion of token rendering wait time) by 71.0% compared to the retransmission method commonly used by real chatbot applications and by 31.6% compared to the baseline packet duplication scheme. By tailoring Eloquent to fit the token-by-token generation of LLM, we enable the Chatbots to respond like an eloquent speaker for users to better enjoy pervasive AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11007v1">Threat Modelling and Risk Analysis for Large Language Model (LLM)-Powered Applications</a></div>
    <div class="paper-meta">
      📅 2024-06-16
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized various applications by providing advanced natural language processing capabilities. However, this innovation introduces new cybersecurity challenges. This paper explores the threat modeling and risk analysis specifically tailored for LLM-powered applications. Focusing on potential attacks like data poisoning, prompt injection, SQL injection, jailbreaking, and compositional injection, we assess their impact on security and propose mitigation strategies. We introduce a framework combining STRIDE and DREAD methodologies for proactive threat identification and risk assessment. Furthermore, we examine the feasibility of an end-to-end threat model through a case study of a custom-built LLM-powered application. This model follows Shostack's Four Question Framework, adjusted for the unique threats LLMs present. Our goal is to propose measures that enhance the security of these powerful AI tools, thwarting attacks, and ensuring the reliability and integrity of LLM-integrated systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10977v1">Toward Optimal LLM Alignments Using Two-Player Games</a></div>
    <div class="paper-meta">
      📅 2024-06-16
      | 💬 Our code is released at https://github.com/ruizheng20/gpo
    </div>
    <details class="paper-abstract">
      The standard Reinforcement Learning from Human Feedback (RLHF) framework primarily focuses on optimizing the performance of large language models using pre-collected prompts. However, collecting prompts that provide comprehensive coverage is both tedious and challenging, and often fails to include scenarios that LLMs need to improve on the most. In this paper, we investigate alignment through the lens of two-agent games, involving iterative interactions between an adversarial and a defensive agent. The adversarial agent's task at each step is to generate prompts that expose the weakness of the defensive agent. In return, the defensive agent seeks to improve its responses to these newly identified prompts it struggled with, based on feedback from the reward model. We theoretically demonstrate that this iterative reinforcement learning optimization converges to a Nash Equilibrium for the game induced by the agents. Experimental results in safety scenarios demonstrate that learning in such a competitive environment not only fully trains agents but also leads to policies with enhanced generalization capabilities for both adversarial and defensive agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10903v1">New Solutions on LLM Acceleration, Optimization, and Application</a></div>
    <div class="paper-meta">
      📅 2024-06-16
      | 💬 This is an expanded and more comprehensive study based on our invited DAC-24 paper with the same title and co-authors
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become extremely potent instruments with exceptional capacities for comprehending and producing human-like text in a wide range of applications. However, the increasing size and complexity of LLMs present significant challenges in both training and deployment, leading to substantial computational and storage costs as well as heightened energy consumption. In this paper, we provide a review of recent advancements and research directions aimed at addressing these challenges and enhancing the efficiency of LLM-based systems. We begin by discussing algorithm-level acceleration techniques focused on optimizing LLM inference speed and resource utilization. We also explore LLM-hardware co-design strategies with a vision to improve system efficiency by tailoring hardware architectures to LLM requirements. Further, we delve into LLM-to-accelerator compilation approaches, which involve customizing hardware accelerators for efficient LLM deployment. Finally, as a case study to leverage LLMs for assisting circuit design, we examine LLM-aided design methodologies for an important task: High-Level Synthesis (HLS) functional verification, by creating a new dataset that contains a large number of buggy and bug-free codes, which can be essential for training LLMs to specialize on HLS verification and debugging. For each aspect mentioned above, we begin with a detailed background study, followed by the presentation of several novel solutions proposed to overcome specific challenges. We then outline future research directions to drive further advancements. Through these efforts, we aim to pave the way for more efficient and scalable deployment of LLMs across a diverse range of applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10834v1">Exposing the Achilles' Heel: Evaluating LLMs Ability to Handle Mistakes in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-06-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been applied to Math Word Problems (MWPs) with transformative impacts, revolutionizing how these complex problems are approached and solved in various domains including educational settings. However, the evaluation of these models often prioritizes final accuracy, overlooking the crucial aspect of reasoning capabilities. This work addresses this gap by focusing on the ability of LLMs to detect and correct reasoning mistakes. We introduce a novel dataset MWP-MISTAKE, incorporating MWPs with both correct and incorrect reasoning steps generated through rule-based methods and smaller language models. Our comprehensive benchmarking reveals significant insights into the strengths and weaknesses of state-of-the-art models, such as GPT-4o, GPT-4, GPT-3.5Turbo, and others. We highlight GPT-$o's superior performance in mistake detection and rectification and the persistent challenges faced by smaller models. Additionally, we identify issues related to data contamination and memorization, impacting the reliability of LLMs in real-world applications. Our findings emphasize the importance of rigorous evaluation of reasoning processes and propose future directions to enhance the generalization and robustness of LLMs in mathematical problem-solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10819v1">GUI-WORLD: A Dataset for GUI-oriented Multimodal LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-06-16
    </div>
    <details class="paper-abstract">
      Recently, Multimodal Large Language Models (MLLMs) have been used as agents to control keyboard and mouse inputs by directly perceiving the Graphical User Interface (GUI) and generating corresponding code. However, current agents primarily exhibit excellent understanding capabilities in static environments and are predominantly applied in relatively simple domains, such as Web or mobile interfaces. We argue that a robust GUI agent should be capable of perceiving temporal information on the GUI, including dynamic Web content and multi-step tasks. Additionally, it should possess a comprehensive understanding of various GUI scenarios, including desktop software and multi-window interactions. To this end, this paper introduces a new dataset, termed GUI-World, which features meticulously crafted Human-MLLM annotations, extensively covering six GUI scenarios and eight types of GUI-oriented questions in three formats. We evaluate the capabilities of current state-of-the-art MLLMs, including ImageLLMs and VideoLLMs, in understanding various types of GUI content, especially dynamic and sequential content. Our findings reveal that ImageLLMs struggle with dynamic GUI content without manually annotated keyframes or operation history. On the other hand, VideoLLMs fall short in all GUI-oriented tasks given the sparse GUI video dataset. Based on GUI-World, we take the initial step of leveraging a fine-tuned VideoLLM as a GUI agent, demonstrating an improved understanding of various GUI tasks. However, due to the limitations in the performance of base LLMs, we conclude that using VideoLLMs as GUI agents remains a significant challenge. We believe our work provides valuable insights for future research in dynamic GUI content understanding. The code and dataset are publicly available at our project homepage: https://gui-world.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10979v2">SportsMetrics: Blending Text and Numerical Data to Understand Information Fusion in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-16
      | 💬 ACL 2024 Long Paper
    </div>
    <details class="paper-abstract">
      Large language models hold significant potential for integrating various data types, such as text documents and database records, for advanced analytics. However, blending text and numerical data presents substantial challenges. LLMs need to process and cross-reference entities and numbers, handle data inconsistencies and redundancies, and develop planning capabilities such as building a working memory for managing complex data queries. In this paper, we introduce four novel tasks centered around sports data analytics to evaluate the numerical reasoning and information fusion capabilities of LLMs. These tasks involve providing LLMs with detailed, play-by-play sports game descriptions, then challenging them with adversarial scenarios such as new game rules, longer durations, scrambled narratives, and analyzing key statistics in game summaries. We conduct extensive experiments on NBA and NFL games to assess the performance of LLMs on these tasks. Our benchmark, SportsMetrics, introduces a new mechanism for assessing LLMs' numerical reasoning and fusion skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12007v1">People will agree what I think: Investigating LLM's False Consensus Effect</a></div>
    <div class="paper-meta">
      📅 2024-06-16
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been widely adopted on interactive systems requiring communications. As the false belief in a model can harm the usability of such systems, LLMs should not have cognitive biases that humans have. Especially psychologists focused on the False Consensus Effect (FCE), which can distract smooth communication by posing false beliefs. However, previous studies have less examined FCE in LLMs thoroughly, which needs more consideration of confounding biases, general situations, and prompt changes. Therefore, in this paper, we conduct two studies to deeply examine the FCE phenomenon in LLMs. In Study 1, we investigate whether LLMs have FCE. In Study 2, we explore how various prompting styles affect the demonstration of FCE. As a result of these studies, we identified that popular LLMs have FCE. Also, the result specifies the conditions when the strength of FCE becomes larger or smaller compared to normal usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.17115v2">How Far Are LLMs from Believable AI? A Benchmark for Evaluating the Believability of Human Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2024-06-15
    </div>
    <details class="paper-abstract">
      In recent years, AI has demonstrated remarkable capabilities in simulating human behaviors, particularly those implemented with large language models (LLMs). However, due to the lack of systematic evaluation of LLMs' simulated behaviors, the believability of LLMs among humans remains ambiguous, i.e., it is unclear which behaviors of LLMs are convincingly human-like and which need further improvements. In this work, we design SimulateBench to evaluate the believability of LLMs when simulating human behaviors. In specific, we evaluate the believability of LLMs based on two critical dimensions: 1) consistency: the extent to which LLMs can behave consistently with the given information of a human to simulate; and 2) robustness: the ability of LLMs' simulated behaviors to remain robust when faced with perturbations. SimulateBench includes 65 character profiles and a total of 8,400 questions to examine LLMs' simulated behaviors. Based on SimulateBench, we evaluate the performances of 10 widely used LLMs when simulating characters. The experimental results reveal that current LLMs struggle to align their behaviors with assigned characters and are vulnerable to perturbations in certain factors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12928v1">Evaluating the Generalization Ability of Quantized LLMs: Benchmark, Analysis, and Toolbox</a></div>
    <div class="paper-meta">
      📅 2024-06-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited exciting progress in multiple scenarios, while the huge computational demands hinder their deployments in lots of real-world applications. As an effective means to reduce memory footprint and inference cost, quantization also faces challenges in performance degradation at low bit-widths. Understanding the impact of quantization on LLM capabilities, especially the generalization ability, is crucial. However, the community's main focus remains on the algorithms and models of quantization, with insufficient attention given to whether the quantized models can retain the strong generalization abilities of LLMs. In this work, we fill this gap by providing a comprehensive benchmark suite for this research topic, including an evaluation system, detailed analyses, and a general toolbox. Specifically, based on the dominant pipeline in LLM quantization, we primarily explore the impact of calibration data distribution on the generalization of quantized LLMs and conduct the benchmark using more than 40 datasets within two main scenarios. Based on this benchmark, we conduct extensive experiments with two well-known LLMs (English and Chinese) and four quantization algorithms to investigate this topic in-depth, yielding several counter-intuitive and valuable findings, e.g., models quantized using a calibration set with the same distribution as the test data are not necessarily optimal. Besides, to facilitate future research, we also release a modular-designed toolbox, which decouples the overall pipeline into several separate components, e.g., base LLM module, dataset module, quantizer module, etc. and allows subsequent researchers to easily assemble their methods through a simple configuration. Our benchmark suite is publicly available at https://github.com/TsingmaoAI/MI-optimize
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10590v1">LLM-Mediated Domain-Specific Voice Agents: The Case of TextileBot</a></div>
    <div class="paper-meta">
      📅 2024-06-15
    </div>
    <details class="paper-abstract">
      Developing domain-specific conversational agents (CAs) has been challenged by the need for extensive domain-focused data. Recent advancements in Large Language Models (LLMs) make them a viable option as a knowledge backbone. LLMs behaviour can be enhanced through prompting, instructing them to perform downstream tasks in a zero-shot fashion (i.e. without training). To this end, we incorporated structural knowledge into prompts and used prompted LLMs to build domain-specific voice-based CAs. We demonstrate this approach for the specific domain of textile circularity in form of the design, development, and evaluation of TextileBot. We present the design and development of the voice agent TextileBot and also the insights from an in-person user study (N=30) evaluating three variations of TextileBots. We analyse the human-agent interactions, combining quantitative and qualitative methods. Our results suggest that participants engaged in multi-turn conversations, and their perceptions of the three variation agents and respective interactions varied demonstrating the effectiveness of our prompt-based LLM approach. We discuss the dynamics of these interactions and their implications for designing future voice-based CAs. The results show that our method's potential for building domain-specific CAs. Furthermore, most participants engaged in multi-turn conversations, and their perceptions of the three voice agents and respective interactions varied demonstrating the effectiveness of our prompt-based LLM approach. We discuss the dynamics of these interactions and their implications for designing future voice-based CAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06326v3">Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching</a></div>
    <div class="paper-meta">
      📅 2024-06-15
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle to provide up-to-date information due to their one-time training and the constantly evolving nature of the world. To keep LLMs current, existing approaches typically involve continued pre-training on new documents. However, they frequently face difficulties in extracting stored knowledge. Motivated by the remarkable success of the Feynman Technique in efficient human learning, we introduce Self-Tuning, a learning framework aimed at improving an LLM's ability to effectively acquire new knowledge from raw documents through self-teaching. Specifically, we develop a Self-Teaching strategy that augments the documents with a set of knowledge-intensive tasks created in a self-supervised manner, focusing on three crucial aspects: memorization, comprehension, and self-reflection. In addition, we introduce three Wiki-Newpages-2023-QA datasets to facilitate an in-depth analysis of an LLM's knowledge acquisition ability concerning memorization, extraction, and reasoning. Extensive experimental results on Llama2 family models reveal that Self-Tuning consistently exhibits superior performance across all knowledge acquisition tasks and excels in preserving previous knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.16595v4">Can LLMs Effectively Leverage Graph Structural Information through Prompts, and Why?</a></div>
    <div class="paper-meta">
      📅 2024-06-15
      | 💬 Accepted to Transactions on Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are gaining increasing attention for their capability to process graphs with rich text attributes, especially in a zero-shot fashion. Recent studies demonstrate that LLMs obtain decent text classification performance on common text-rich graph benchmarks, and the performance can be improved by appending encoded structural information as natural languages into prompts. We aim to understand why the incorporation of structural information inherent in graph data can improve the prediction performance of LLMs. First, we rule out the concern of data leakage by curating a novel leakage-free dataset and conducting a comparative analysis alongside a previously widely-used dataset. Second, as past work usually encodes the ego-graph by describing the graph structure in natural language, we ask the question: do LLMs understand the graph structure in accordance with the intent of the prompt designers? Third, we investigate why LLMs can improve their performance after incorporating structural information. Our exploration of these questions reveals that (i) there is no substantial evidence that the performance of LLMs is significantly attributed to data leakage; (ii) instead of understanding prompts as graph structures as intended by the prompt designers, LLMs tend to process prompts more as contextual paragraphs and (iii) the most efficient elements of the local neighborhood included in the prompt are phrases that are pertinent to the node label, rather than the graph structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07138v3">Unprecedented Code Change Automation: The Fusion of LLMs and Transformation by Example</a></div>
    <div class="paper-meta">
      📅 2024-06-15
      | 💬 This paper is accepted to Proceedings of the 32nd ACM Symposium on the Foundations of Software Engineering (FSE - 2024), This is an author copy
    </div>
    <details class="paper-abstract">
      Software developers often repeat code changes, known as "code change patterns" (CPATs), within and across projects. Automating these CPATs accelerates development, but current Transformation by Example (TBE) techniques are limited by the input examples' quality and quantity, missing variations with different syntax or flow yet semantically similar. Large Language Models (LLMs), trained on vast code datasets, can overcome these limitations by generating semantically equivalent, unseen CPAT variants, enhancing TBE effectiveness. We identified best practices for using LLMs to generate code variants meeting criteria of correctness, usefulness, and applicability. Implementing these in PyCraft, combining static and dynamic analysis with LLMs, we achieved an F-measure of 96.6% in identifying correct variants, expanding inputs by 58x on average, and automating changes to increase target codes by up to 39x. Patches from PyCraft were submitted to projects like microsoft/DeepSpeed and IBM/inFairness, with an 83% acceptance rate, validating our approach's usefulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08269v2">Analyzing constrained LLM through PDFA-learning</a></div>
    <div class="paper-meta">
      📅 2024-06-15
      | 💬 Workshop Paper
    </div>
    <details class="paper-abstract">
      We define a congruence that copes with null next-symbol probabilities that arise when the output of a language model is constrained by some means during text generation. We develop an algorithm for efficiently learning the quotient with respect to this congruence and evaluate it on case studies for analyzing statistical properties of LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06431v2">Human-AI Collaborative Essay Scoring: A Dual-Process Framework with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-15
    </div>
    <details class="paper-abstract">
      Receiving timely and personalized feedback is essential for second-language learners, especially when human instructors are unavailable. This study explores the effectiveness of Large Language Models (LLMs), including both proprietary and open-source models, for Automated Essay Scoring (AES). Through extensive experiments with public and private datasets, we find that while LLMs do not surpass conventional state-of-the-art (SOTA) grading models in performance, they exhibit notable consistency, generalizability, and explainability. We propose an open-source LLM-based AES system, inspired by the dual-process theory. Our system offers accurate grading and high-quality feedback, at least comparable to that of fine-tuned proprietary LLMs, in addition to its ability to alleviate misgrading. Furthermore, we conduct human-AI co-grading experiments with both novice and expert graders. We find that our system not only automates the grading process but also enhances the performance and efficiency of human graders, particularly for essays where the model has lower confidence. These results highlight the potential of LLMs to facilitate effective human-AI collaboration in the educational context, potentially transforming learning experiences through AI-generated feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10774v3">Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</a></div>
    <div class="paper-meta">
      📅 2024-06-14
      | 💬 The code for this implementation is available at https://github.com/FasterDecoding/Medusa
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) employ auto-regressive decoding that requires sequential computation, with each step reliant on the previous one's output. This creates a bottleneck as each step necessitates moving the full model parameters from High-Bandwidth Memory (HBM) to the accelerator's cache. While methods such as speculative decoding have been suggested to address this issue, their implementation is impeded by the challenges associated with acquiring and maintaining a separate draft model. In this paper, we present Medusa, an efficient method that augments LLM inference by adding extra decoding heads to predict multiple subsequent tokens in parallel. Using a tree-based attention mechanism, Medusa constructs multiple candidate continuations and verifies them simultaneously in each decoding step. By leveraging parallel processing, Medusa substantially reduces the number of decoding steps required. We present two levels of fine-tuning procedures for Medusa to meet the needs of different use cases: Medusa-1: Medusa is directly fine-tuned on top of a frozen backbone LLM, enabling lossless inference acceleration. Medusa-2: Medusa is fine-tuned together with the backbone LLM, enabling better prediction accuracy of Medusa heads and higher speedup but needing a special training recipe that preserves the backbone model's capabilities. Moreover, we propose several extensions that improve or expand the utility of Medusa, including a self-distillation to handle situations where no training data is available and a typical acceptance scheme to boost the acceptance rate while maintaining generation quality. We evaluate Medusa on models of various sizes and training procedures. Our experiments demonstrate that Medusa-1 can achieve over 2.2x speedup without compromising generation quality, while Medusa-2 further improves the speedup to 2.3-3.6x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10375v1">Mokav: Execution-driven Differential Testing with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-14
    </div>
    <details class="paper-abstract">
      It is essential to detect functional differences in various software engineering tasks, such as automated program repair, mutation testing, and code refactoring. The problem of detecting functional differences between two programs can be reduced to searching for a difference exposing test (DET): a test input that results in different outputs on the subject programs. In this paper, we propose Mokav, a novel execution-driven tool that leverages LLMs to generate DETs. Mokav takes two versions of a program (P and Q) and an example test input. When successful, Mokav generates a valid DET, a test input that leads to different outputs on P and Q. Mokav iteratively prompts an LLM with a specialized prompt to generate new test inputs. At each iteration, Mokav provides execution-based feedback regarding previously generated tests until the LLM produces a DET. We evaluate Mokav on 1,535 pairs of Python programs collected from the Codeforces competition platform and 32 pairs of programs from the QuixBugs dataset. Our experiments show that Mokav outperforms the state-of-the-art, Pynguin and Differential Prompting, by a large margin. Mokav can generate DETs for 81.7% (1,255/1,535) of the program pairs in our benchmark (versus 4.9% for Pynguin and 37.3% for Differential Prompting). We demonstrate that all components in our system, including the iterative and execution-driven approaches, contribute to its high effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10181v1">Practical offloading for fine-tuning LLM on commodity GPU via learned subspace projectors</a></div>
    <div class="paper-meta">
      📅 2024-06-14
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) requires significant memory, often exceeding the capacity of a single GPU. A common solution to this memory challenge is offloading compute and data from the GPU to the CPU. However, this approach is hampered by the limited bandwidth of commodity hardware, which constrains communication between the CPU and GPU. In this paper, we present an offloading framework, LSP_Offload, that enables near-native speed LLM fine-tuning on commodity hardware through learned subspace projectors. Our data-driven approach involves learning an efficient sparse compressor that minimizes communication with minimal precision loss. Additionally, we introduce a novel layer-wise communication schedule to maximize parallelism between communication and computation. As a result, our framework can fine-tune a 1.3 billion parameter model on a 4GB laptop GPU and a 7 billion parameter model on an NVIDIA RTX 4090 GPU with 24GB memory, achieving only a 31% slowdown compared to fine-tuning with unlimited memory. Compared to state-of-the-art offloading frameworks, our approach increases fine-tuning throughput by up to 3.33 times and reduces end-to-end fine-tuning time by 33.1%~62.5% when converging to the same accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10018v1">STALL+: Boosting LLM-based Repository-level Code Completion with Static Analysis</a></div>
    <div class="paper-meta">
      📅 2024-06-14
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Repository-level code completion is challenging as it involves complicated contexts from multiple files in the repository. To date, researchers have proposed two technical categories to enhance LLM-based repository-level code completion, i.e., retrieval-augmented generation (RAG) and static analysis integration. This work performs the first study on the static analysis integration in LLM-based repository-level code completion by investigating both the effectiveness and efficiency of static analysis integration strategies across different phases of code completion. We first implement a framework STALL+, which supports an extendable and customizable integration of multiple static analysis strategies into the complete pipeline of LLM-based repository-level code completion; and based on STALL+, we perform extensive experiments by including different code LLMs on the latest repository-level code completion benchmark CrossCodeEval. Our findings show that integrating file-level dependencies in prompting phase performs the best while the integration in post-processing phase performs the worse. Additionally, we observe different improvements from static analysis between dynamic languages and static languages, i.e., the best combination is prompting-phase with decoding-phase integration for Java while the best combination is prompting-phase with post-processing-phase integration for Python given the limitations of statically analyzing dynamic languages. Additionally, we find the complementarity between RAG and static analysis integration as well as their cost-effectiveness after combination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09972v1">A Better LLM Evaluator for Text Generation: The Impact of Prompt Output Sequencing and Optimization</a></div>
    <div class="paper-meta">
      📅 2024-06-14
      | 💬 Presented in JSAI 2024. The first two authors contributed equally. arXiv admin note: substantial text overlap with arXiv:2406.02863
    </div>
    <details class="paper-abstract">
      This research investigates prompt designs of evaluating generated texts using large language models (LLMs). While LLMs are increasingly used for scoring various inputs, creating effective prompts for open-ended text evaluation remains challenging due to model sensitivity and subjectivity in evaluation of text generation. Our study experimented with different prompt structures, altering the sequence of output instructions and including explanatory reasons. We found that the order of presenting reasons and scores significantly influences LLMs' scoring, with a different level of rule understanding in the prompt. An additional optimization may enhance scoring alignment if sufficient data is available. This insight is crucial for improving the accuracy and consistency of LLM-based evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10434v2">Using an LLM to Turn Sign Spottings into Spoken Language Sentences</a></div>
    <div class="paper-meta">
      📅 2024-06-14
    </div>
    <details class="paper-abstract">
      Sign Language Translation (SLT) is a challenging task that aims to generate spoken language sentences from sign language videos. In this paper, we introduce a hybrid SLT approach, Spotter+GPT, that utilizes a sign spotter and a powerful Large Language Model (LLM) to improve SLT performance. Spotter+GPT breaks down the SLT task into two stages. The videos are first processed by the Spotter, which is trained on a linguistic sign language dataset, to identify individual signs. These spotted signs are then passed to an LLM, which transforms them into coherent and contextually appropriate spoken language sentences. The source code of the Spotter is available at https://gitlab.surrey.ac.uk/cogvispublic/sign-spotter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15529v2">LimGen: Probing the LLMs for Generating Suggestive Limitations of Research Papers</a></div>
    <div class="paper-meta">
      📅 2024-06-14
      | 💬 Accepted at ECML-PKDD 2024
    </div>
    <details class="paper-abstract">
      Examining limitations is a crucial step in the scholarly research reviewing process, revealing aspects where a study might lack decisiveness or require enhancement. This aids readers in considering broader implications for further research. In this article, we present a novel and challenging task of Suggestive Limitation Generation (SLG) for research papers. We compile a dataset called \textbf{\textit{LimGen}}, encompassing 4068 research papers and their associated limitations from the ACL anthology. We investigate several approaches to harness large language models (LLMs) for producing suggestive limitations, by thoroughly examining the related challenges, practical insights, and potential opportunities. Our LimGen dataset and code can be accessed at \url{https://github.com/arbmf/LimGen}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08183v2">Underneath the Numbers: Quantitative and Qualitative Gender Fairness in LLMs for Depression Prediction</a></div>
    <div class="paper-meta">
      📅 2024-06-14
    </div>
    <details class="paper-abstract">
      Recent studies show bias in many machine learning models for depression detection, but bias in LLMs for this task remains unexplored. This work presents the first attempt to investigate the degree of gender bias present in existing LLMs (ChatGPT, LLaMA 2, and Bard) using both quantitative and qualitative approaches. From our quantitative evaluation, we found that ChatGPT performs the best across various performance metrics and LLaMA 2 outperforms other LLMs in terms of group fairness metrics. As qualitative fairness evaluation remains an open research question we propose several strategies (e.g., word count, thematic analysis) to investigate whether and how a qualitative evaluation can provide valuable insights for bias analysis beyond what is possible with quantitative evaluation. We found that ChatGPT consistently provides a more comprehensive, well-reasoned explanation for its prediction compared to LLaMA 2. We have also identified several themes adopted by LLMs to qualitatively evaluate gender fairness. We hope our results can be used as a stepping stone towards future attempts at improving qualitative evaluation of fairness for LLMs especially for high-stakes tasks such as depression detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15126v1">On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-06-14
      | 💬 A survey on LLMs-driven synthetic data generation, curation and evaluation
    </div>
    <details class="paper-abstract">
      Within the evolving landscape of deep learning, the dilemma of data quantity and quality has been a long-standing problem. The recent advent of Large Language Models (LLMs) offers a data-centric solution to alleviate the limitations of real-world data with synthetic data generation. However, current investigations into this field lack a unified framework and mostly stay on the surface. Therefore, this paper provides an organization of relevant studies based on a generic workflow of synthetic data generation. By doing so, we highlight the gaps within existing research and outline prospective avenues for future study. This work aims to shepherd the academic and industrial communities towards deeper, more methodical inquiries into the capabilities and applications of LLMs-driven synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10300v1">Large Language Models as Software Components: A Taxonomy for LLM-Integrated Applications</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become widely adopted recently. Research explores their use both as autonomous agents and as tools for software engineering. LLM-integrated applications, on the other hand, are software systems that leverage an LLM to perform tasks that would otherwise be impossible or require significant coding effort. While LLM-integrated application engineering is emerging as new discipline, its terminology, concepts and methods need to be established. This study provides a taxonomy for LLM-integrated applications, offering a framework for analyzing and describing these systems. It also demonstrates various ways to utilize LLMs in applications, as well as options for implementing such integrations. Following established methods, we analyze a sample of recent LLM-integrated applications to identify relevant dimensions. We evaluate the taxonomy by applying it to additional cases. This review shows that applications integrate LLMs in numerous ways for various purposes. Frequently, they comprise multiple LLM integrations, which we term ``LLM components''. To gain a clear understanding of an application's architecture, we examine each LLM component separately. We identify thirteen dimensions along which to characterize an LLM component, including the LLM skills leveraged, the format of the output, and more. LLM-integrated applications are described as combinations of their LLM components. We suggest a concise representation using feature vectors for visualization. The taxonomy is effective for describing LLM-integrated applications. It can contribute to theory building in the nascent field of LLM-integrated application engineering and aid in developing such systems. Researchers and practitioners explore numerous creative ways to leverage LLMs in applications. Though challenges persist, integrating LLMs may revolutionize the way software systems are built.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09569v1">Speech ReaLLM -- Real-time Streaming Speech Recognition with Multimodal LLMs by Teaching the Flow of Time</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      We introduce Speech ReaLLM, a new ASR architecture that marries "decoder-only" ASR with the RNN-T to make multimodal LLM architectures capable of real-time streaming. This is the first "decoder-only" ASR architecture designed to handle continuous audio without explicit end-pointing. Speech ReaLLM is a special case of the more general ReaLLM ("real-time LLM") approach, also introduced here for the first time. The idea is inspired by RNN-T: Instead of generating a response only at the end of a user prompt, generate after every input token received in real time (it is often empty). On Librispeech "test", an 80M Speech ReaLLM achieves WERs of 3.0% and 7.4% in real time (without an external LM or auxiliary loss). This is only slightly above a 3x larger Attention-Encoder-Decoder baseline. We also show that this way, an LLM architecture can learn to represent and reproduce the flow of time; and that a pre-trained 7B LLM can be fine-tuned to do reasonably well on this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00343v2">Beyond Metrics: Evaluating LLMs' Effectiveness in Culturally Nuanced, Low-Resource Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) in real-world applications presents both opportunities and challenges, particularly in multilingual and code-mixed communication settings. This research evaluates the performance of seven leading LLMs in sentiment analysis on a dataset derived from multilingual and code-mixed WhatsApp chats, including Swahili, English and Sheng. Our evaluation includes both quantitative analysis using metrics like F1 score and qualitative assessment of LLMs' explanations for their predictions. We find that, while Mistral-7b and Mixtral-8x7b achieved high F1 scores, they and other LLMs such as GPT-3.5-Turbo, Llama-2-70b, and Gemma-7b struggled with understanding linguistic and contextual nuances, as well as lack of transparency in their decision-making process as observed from their explanations. In contrast, GPT-4 and GPT-4-Turbo excelled in grasping diverse linguistic inputs and managing various contextual information, demonstrating high consistency with human alignment and transparency in their decision-making process. The LLMs however, encountered difficulties in incorporating cultural nuance especially in non-English settings with GPT-4s doing so inconsistently. The findings emphasize the necessity of continuous improvement of LLMs to effectively tackle the challenges of culturally nuanced, low-resource real-world settings and the need for developing evaluation benchmarks for capturing these issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09233v1">C2HLSC: Can LLMs Bridge the Software-to-Hardware Design Gap?</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 Accepted at The First IEEE International Workshop on LLM-Aided Design
    </div>
    <details class="paper-abstract">
      High Level Synthesis (HLS) tools offer rapid hardware design from C code, but their compatibility is limited by code constructs. This paper investigates Large Language Models (LLMs) for refactoring C code into HLS-compatible formats. We present several case studies by using an LLM to rewrite C code for NIST 800-22 randomness tests, a QuickSort algorithm and AES-128 into HLS-synthesizable c. The LLM iteratively transforms the C code guided by user prompts, implementing functions like streaming data and hardware-specific signals. This evaluation demonstrates the LLM's potential to assist hardware design refactoring regular C code into HLS synthesizable C code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03339v2">The Challenges of Evaluating LLM Applications: An Analysis of Automated, Human, and LLM-Based Approaches</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 Accepted in The First Workshop on Large Language Models for Evaluation in Information Retrieval
    </div>
    <details class="paper-abstract">
      Chatbots have been an interesting application of natural language generation since its inception. With novel transformer based Generative AI methods, building chatbots have become trivial. Chatbots which are targeted at specific domains for example medicine and psychology are implemented rapidly. This however, should not distract from the need to evaluate the chatbot responses. Especially because the natural language generation community does not entirely agree upon how to effectively evaluate such applications. With this work we discuss the issue further with the increasingly popular LLM based evaluations and how they correlate with human evaluations. Additionally, we introduce a comprehensive factored evaluation mechanism that can be utilized in conjunction with both human and LLM-based evaluations. We present the results of an experimental evaluation conducted using this scheme in one of our chatbot implementations which consumed educational reports, and subsequently compare automated, traditional human evaluation, factored human evaluation, and factored LLM evaluation. Results show that factor based evaluation produces better insights on which aspects need to be improved in LLM applications and further strengthens the argument to use human evaluation in critical spaces where main functionality is not direct retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09187v1">GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has catalyzed the deployment of LLM-powered agents across numerous applications, raising new concerns regarding their safety and trustworthiness. Existing methods for enhancing the safety of LLMs are not directly transferable to LLM-powered agents due to their diverse objectives and output modalities. In this paper, we propose GuardAgent, the first LLM agent as a guardrail to other LLM agents. Specifically, GuardAgent oversees a target LLM agent by checking whether its inputs/outputs satisfy a set of given guard requests defined by the users. GuardAgent comprises two steps: 1) creating a task plan by analyzing the provided guard requests, and 2) generating guardrail code based on the task plan and executing the code by calling APIs or using external engines. In both steps, an LLM is utilized as the core reasoning component, supplemented by in-context demonstrations retrieved from a memory module. Such knowledge-enabled reasoning allows GuardAgent to understand various textual guard requests and accurately "translate" them into executable code that provides reliable guardrails. Furthermore, GuardAgent is equipped with an extendable toolbox containing functions and APIs and requires no additional LLM training, which underscores its generalization capabilities and low operational overhead. Additionally, we propose two novel benchmarks: an EICU-AC benchmark for assessing privacy-related access control for healthcare agents and a Mind2Web-SC benchmark for safety evaluation for web agents. We show the effectiveness of GuardAgent on these two benchmarks with 98.7% and 90.0% accuracy in moderating invalid inputs and outputs for the two types of agents, respectively. We also show that GuardAgent is able to define novel functions in adaption to emergent LLM agents and guard requests, which underscores its strong generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09170v1">Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have showcased remarkable reasoning capabilities, yet they remain susceptible to errors, particularly in temporal reasoning tasks involving complex temporal logic. Existing research has explored LLM performance on temporal reasoning using diverse datasets and benchmarks. However, these studies often rely on real-world data that LLMs may have encountered during pre-training or employ anonymization techniques that can inadvertently introduce factual inconsistencies. In this work, we address these limitations by introducing novel synthetic datasets specifically designed to assess LLM temporal reasoning abilities in various scenarios. The diversity of question types across these datasets enables systematic investigation into the impact of the problem structure, size, question type, fact order, and other factors on LLM performance. Our findings provide valuable insights into the strengths and weaknesses of current LLMs in temporal reasoning tasks. To foster further research in this area, we are open-sourcing the datasets and evaluation framework used in our experiments: https://huggingface.co/datasets/baharef/ToT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09155v1">DefAn: Definitive Answer Dataset for LLMs Hallucination Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities, revolutionizing the integration of AI in daily life applications. However, they are prone to hallucinations, generating claims that contradict established facts, deviating from prompts, and producing inconsistent responses when the same prompt is presented multiple times. Addressing these issues is challenging due to the lack of comprehensive and easily assessable benchmark datasets. Most existing datasets are small and rely on multiple-choice questions, which are inadequate for evaluating the generative prowess of LLMs. To measure hallucination in LLMs, this paper introduces a comprehensive benchmark dataset comprising over 75,000 prompts across eight domains. These prompts are designed to elicit definitive, concise, and informative answers. The dataset is divided into two segments: one publicly available for testing and assessing LLM performance and a hidden segment for benchmarking various LLMs. In our experiments, we tested six LLMs-GPT-3.5, LLama 2, LLama 3, Gemini, Mixtral, and Zephyr-revealing that overall factual hallucination ranges from 59% to 82% on the public dataset and 57% to 76% in the hidden benchmark. Prompt misalignment hallucination ranges from 6% to 95% in the public dataset and 17% to 94% in the hidden counterpart. Average consistency ranges from 21% to 61% and 22% to 63%, respectively. Domain-wise analysis shows that LLM performance significantly deteriorates when asked for specific numeric information while performing moderately with person, location, and date queries. Our dataset demonstrates its efficacy and serves as a comprehensive benchmark for LLM performance evaluation. Our dataset and LLMs responses are available at \href{https://github.com/ashikiut/DefAn}{https://github.com/ashikiut/DefAn}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.08144v4">Mobile-Env: Building Qualified Evaluation Benchmarks for LLM-GUI Interaction</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      The Graphical User Interface (GUI) is pivotal for human interaction with the digital world, enabling efficient device control and the completion of complex tasks. Recent progress in Large Language Models (LLMs) and Vision Language Models (VLMs) offers the chance to create advanced GUI agents. To ensure their effectiveness, there's a pressing need for qualified benchmarks that provide trustworthy and reproducible evaluations -- a challenge current benchmarks often fail to address. To tackle this issue, we introduce Mobile-Env, a comprehensive toolkit tailored for creating GUI benchmarks in the Android mobile environment. Mobile-Env offers an isolated and controllable setting for reliable evaluations, and accommodates intermediate instructions and rewards to reflect real-world usage more naturally. Utilizing Mobile-Env, we collect an open-world task set across various real-world apps and a fixed world set, WikiHow, which captures a significant amount of dynamic online contents for fully controllable and reproducible evaluation. We conduct comprehensive evaluations of LLM agents using these benchmarks. Our findings reveal that even advanced models (e.g., GPT-4V and LLaMA-3) struggle with tasks that are relatively simple for humans. This highlights a crucial gap in current models and underscores the importance of developing more capable foundation models and more effective GUI agent frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09012v1">Bayesian Statistical Modeling with Predictors from LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 20 pages, 10 figures, parallel submission to a journal
    </div>
    <details class="paper-abstract">
      State of the art large language models (LLMs) have shown impressive performance on a variety of benchmark tasks and are increasingly used as components in larger applications, where LLM-based predictions serve as proxies for human judgements or decision. This raises questions about the human-likeness of LLM-derived information, alignment with human intuition, and whether LLMs could possibly be considered (parts of) explanatory models of (aspects of) human cognition or language use. To shed more light on these issues, we here investigate the human-likeness of LLMs' predictions for multiple-choice decision tasks from the perspective of Bayesian statistical modeling. Using human data from a forced-choice experiment on pragmatic language use, we find that LLMs do not capture the variance in the human data at the item-level. We suggest different ways of deriving full distributional predictions from LLMs for aggregate, condition-level data, and find that some, but not all ways of obtaining condition-level predictions yield adequate fits to human data. These results suggests that assessment of LLM performance depends strongly on seemingly subtle choices in methodology, and that LLMs are at best predictors of human behavior at the aggregate, condition-level, for which they are, however, not designed to, or usually used to, make predictions in the first place.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10295v1">Robustness of Structured Data Extraction from In-plane Rotated Documents using Multi-Modal Large Language Models (LLM)</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Multi-modal large language models (LLMs) have shown remarkable performance in various natural language processing tasks, including data extraction from documents. However, the accuracy of these models can be significantly affected by document in-plane rotation, also known as skew, a common issue in real-world scenarios for scanned documents. This study investigates the impact of document skew on the data extraction accuracy of three state-of-the-art multi-modal LLMs: Anthropic Claude V3 Sonnet, GPT-4-Turbo, and Llava:v1.6. We focus on extracting specific entities from synthetically generated sample documents with varying degrees of skewness. The results demonstrate that document skew adversely affects the data extraction accuracy of all the tested LLMs, with the severity of the impact varying across models. We identify the safe in-plane rotation angles (SIPRA) for each model and investigate the effects of skew on model hallucinations. Furthermore, we explore existing skew detection and correction mechanisms and discuss their potential limitations. We propose alternative approaches, including developing new multi-modal architectures that are inherently more robust to document skew and incorporating skewing techniques during the pre-training phase of the models. Additionally, we highlight the need for more comprehensive testing on a wider range of document quality and conditions to fully understand the challenges and opportunities associated with using multi-modal LLMs for information extraction in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05839v2">MaLa-ASR: Multimedia-Assisted LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      As more and more information-rich data like video become available, utilizing multi-modal auxiliary information to enhance audio tasks has sparked widespread research interest. The recent surge in research on LLM-based audio models provides fresh perspectives for tackling audio tasks. Given that LLM can flexibly ingest multiple inputs, we propose MaLa-ASR, an LLM-based ASR model that can integrate textual keywords extracted from presentation slides to improve recognition of conference content. MaLa-ASR yields average WERs of 9.4% and 11.7% on the L95 and S95 subsets of the SlideSpeech corpus, representing a significant relative WER drop of 27.9% and 44.7% over the baseline model reported in SlideSpeech. MaLa-ASR underscores LLM's strong performance in speech tasks and the capability to integrate auxiliary information conveniently. By adding keywords to the input prompt, the biased word error rate (B-WER) reduces relatively by 46.0% and 44.2%, establishing a new SOTA on this dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05644v2">How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) rely on safety alignment to avoid responding to malicious user inputs. Unfortunately, jailbreak can circumvent safety guardrails, resulting in LLMs generating harmful content and raising concerns about LLM safety. Due to language models with intensive parameters often regarded as black boxes, the mechanisms of alignment and jailbreak are challenging to elucidate. In this paper, we employ weak classifiers to explain LLM safety through the intermediate hidden states. We first confirm that LLMs learn ethical concepts during pre-training rather than alignment and can identify malicious and normal inputs in the early layers. Alignment actually associates the early concepts with emotion guesses in the middle layers and then refines them to the specific reject tokens for safe generations. Jailbreak disturbs the transformation of early unethical classification into negative emotions. We conduct experiments on models from 7B to 70B across various model families to prove our conclusion. Overall, our paper indicates the intrinsical mechanism of LLM safety and how jailbreaks circumvent safety guardrails, offering a new perspective on LLM safety and reducing concerns. Our code is available at https://github.com/ydyjya/LLM-IHS-Explanation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08824v1">LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 40 pages (52 with references), 21 Figures, 6 Tables
    </div>
    <details class="paper-abstract">
      Members of the Human-Robot Interaction (HRI) and Artificial Intelligence (AI) communities have proposed Large Language Models (LLMs) as a promising resource for robotics tasks such as natural language interactions, doing household and workplace tasks, approximating `common sense reasoning', and modeling humans. However, recent research has raised concerns about the potential for LLMs to produce discriminatory outcomes and unsafe behaviors in real-world robot experiments and applications. To address these concerns, we conduct an HRI-based evaluation of discrimination and safety criteria on several highly-rated LLMs. Our evaluation reveals that LLMs currently lack robustness when encountering people across a diverse range of protected identity characteristics (e.g., race, gender, disability status, nationality, religion, and their intersections), producing biased outputs consistent with directly discriminatory outcomes -- e.g. `gypsy' and `mute' people are labeled untrustworthy, but not `european' or `able-bodied' people. Furthermore, we test models in settings with unconstrained natural language (open vocabulary) inputs, and find they fail to act safely, generating responses that accept dangerous, violent, or unlawful instructions -- such as incident-causing misstatements, taking people's mobility aids, and sexual predation. Our results underscore the urgent need for systematic, routine, and comprehensive risk assessments and assurances to improve outcomes and ensure LLMs only operate on robots when it is safe, effective, and just to do so. Data and code will be made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08802v1">DubWise: Video-Guided Speech Duration Control in Multimodal LLM-based Text-to-Speech for Dubbing</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 Accepted at INTERSPEECH 2024
    </div>
    <details class="paper-abstract">
      Audio-visual alignment after dubbing is a challenging research problem. To this end, we propose a novel method, DubWise Multi-modal Large Language Model (LLM)-based Text-to-Speech (TTS), which can control the speech duration of synthesized speech in such a way that it aligns well with the speakers lip movements given in the reference video even when the spoken text is different or in a different language. To accomplish this, we propose to utilize cross-modal attention techniques in a pre-trained GPT-based TTS. We combine linguistic tokens from text, speaker identity tokens via a voice cloning network, and video tokens via a proposed duration controller network. We demonstrate the effectiveness of our system on the Lip2Wav-Chemistry and LRS2 datasets. Also, the proposed method achieves improved lip sync and naturalness compared to the SOTAs for the same language but different text (i.e., non-parallel) and the different language, different text (i.e., cross-lingual) scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09464v1">GPT-ology, Computational Models, Silicon Sampling: How should we think about LLMs in Cognitive Science?</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 CogSci 2024; 6 pages + 2 page of references
    </div>
    <details class="paper-abstract">
      Large Language Models have taken the cognitive science world by storm. It is perhaps timely now to take stock of the various research paradigms that have been used to make scientific inferences about ``cognition" in these models or about human cognition. We review several emerging research paradigms -- GPT-ology, LLMs-as-computational-models, and ``silicon sampling" -- and review recent papers that have used LLMs under these paradigms. In doing so, we discuss their claims as well as challenges to scientific inference under these various paradigms. We highlight several outstanding issues about LLMs that have to be addressed to push our science forward: closed-source vs open-sourced models; (the lack of visibility of) training data; and reproducibility in LLM research, including forming conventions on new task ``hyperparameters" like instructions and prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10291v1">ResearchArena: Benchmarking LLMs' Ability to Collect and Organize Information as Research Agents</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited remarkable performance across various tasks in natural language processing. Nevertheless, challenges still arise when these tasks demand domain-specific expertise and advanced analytical skills, such as conducting research surveys on a designated topic. In this research, we develop ResearchArena, a benchmark that measures LLM agents' ability to conduct academic surveys, an initial step of academic research process. Specifically, we deconstructs the surveying process into three stages 1) information discovery: locating relevant papers, 2) information selection: assessing papers' importance to the topic, and 3) information organization: organizing papers into meaningful structures. In particular, we establish an offline environment comprising 12.0M full-text academic papers and 7.9K survey papers, which evaluates agents' ability to locate supporting materials for composing the survey on a topic, rank the located papers based on their impact, and organize these into a hierarchical knowledge mind-map. With this benchmark, we conduct preliminary evaluations of existing techniques and find that all LLM-based methods under-performing when compared to basic keyword-based retrieval techniques, highlighting substantial opportunities for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02015v2">MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance, and organizations are racing to serve LLMs of varying sizes as endpoints for use-cases like chat, programming and search. However, efficiently serving multiple LLMs poses significant challenges for existing approaches due to varying popularity of LLMs. In the paper, we present MuxServe, a flexible spatial-temporal multiplexing system for efficient multiple LLM serving. The key insight behind is to colocate LLMs considering their popularity to multiplex memory resources, and leverage the characteristics of prefill and decoding phases to separate and flexibly colocate them to multiplex computation resources. MuxServe formally formulates the multiplexing problem, and proposes a novel placement algorithm and adaptive batch scheduling strategy to identify optimal colocations and maximize utilization. MuxServe designs a unified resource manager to enable flexible and efficient multiplexing. Evaluation results show that MuxServe can achieves up to $1.8\times$ higher throughput or processes $2.9\times$ more requests within $99\%$ SLO attainment. The code is available at: \url{https://github.com/hao-ai-lab/MuxServe}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08765v1">LLM-based Knowledge Pruning for Time Series Data Analytics on Edge-computing Devices</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Limited by the scale and diversity of time series data, the neural networks trained on time series data often overfit and show unsatisfacotry performances. In comparison, large language models (LLMs) recently exhibit impressive generalization in diverse fields. Although massive LLM based approaches are proposed for time series tasks, these methods require to load the whole LLM in both training and reference. This high computational demands limit practical applications in resource-constrained settings, like edge-computing and IoT devices. To address this issue, we propose Knowledge Pruning (KP), a novel paradigm for time series learning in this paper. For a specific downstream task, we argue that the world knowledge learned by LLMs is much redundant and only the related knowledge termed as "pertinent knowledge" is useful. Unlike other methods, our KP targets to prune the redundant knowledge and only distill the pertinent knowledge into the target model. This reduces model size and computational costs significantly. Additionally, different from existing LLM based approaches, our KP does not require to load the LLM in the process of training and testing, further easing computational burdens. With our proposed KP, a lightweight network can effectively learn the pertinent knowledge, achieving satisfactory performances with a low computation cost. To verify the effectiveness of our KP, two fundamental tasks on edge-computing devices are investigated in our experiments, where eight diverse environments or benchmarks with different networks are used to verify the generalization of our KP. Through experiments, our KP demonstrates effective learning of pertinent knowledge, achieving notable performance improvements in regression (19.7% on average) and classification (up to 13.7%) tasks, showcasing state-of-the-art results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08725v1">RL-JACK: Reinforcement Learning-powered Black-box Jailbreaking Attack against LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      Modern large language model (LLM) developers typically conduct a safety alignment to prevent an LLM from generating unethical or harmful content. Recent studies have discovered that the safety alignment of LLMs can be bypassed by jailbreaking prompts. These prompts are designed to create specific conversation scenarios with a harmful question embedded. Querying an LLM with such prompts can mislead the model into responding to the harmful question. The stochastic and random nature of existing genetic methods largely limits the effectiveness and efficiency of state-of-the-art (SOTA) jailbreaking attacks. In this paper, we propose RL-JACK, a novel black-box jailbreaking attack powered by deep reinforcement learning (DRL). We formulate the generation of jailbreaking prompts as a search problem and design a novel RL approach to solve it. Our method includes a series of customized designs to enhance the RL agent's learning efficiency in the jailbreaking context. Notably, we devise an LLM-facilitated action space that enables diverse action variations while constraining the overall search space. We propose a novel reward function that provides meaningful dense rewards for the agent toward achieving successful jailbreaking. Through extensive evaluations, we demonstrate that RL-JACK is overall much more effective than existing jailbreaking attacks against six SOTA LLMs, including large open-source models and commercial models. We also show the RL-JACK's resiliency against three SOTA defenses and its transferability across different models. Finally, we validate the insensitivity of RL-JACK to the variations in key hyper-parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.01817v3">LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      There is considerable confusion about the role of Large Language Models (LLMs) in planning and reasoning tasks. On one side are over-optimistic claims that LLMs can indeed do these tasks with just the right prompting or self-verification strategies. On the other side are perhaps over-pessimistic claims that all that LLMs are good for in planning/reasoning tasks are as mere translators of the problem specification from one syntactic format to another, and ship the problem off to external symbolic solvers. In this position paper, we take the view that both these extremes are misguided. We argue that auto-regressive LLMs cannot, by themselves, do planning or self-verification (which is after all a form of reasoning), and shed some light on the reasons for misunderstandings in the literature. We will also argue that LLMs should be viewed as universal approximate knowledge sources that have much more meaningful roles to play in planning/reasoning tasks beyond simple front-end/back-end format translators. We present a vision of {\bf LLM-Modulo Frameworks} that combine the strengths of LLMs with external model-based verifiers in a tighter bi-directional interaction regime. We will show how the models driving the external verifiers themselves can be acquired with the help of LLMs. We will also argue that rather than simply pipelining LLMs and symbolic components, this LLM-Modulo Framework provides a better neuro-symbolic approach that offers tighter integration between LLMs and symbolic components, and allows extending the scope of model-based planning/reasoning regimes towards more flexible knowledge, problem and preference specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10290v1">MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) and Large Multimodal Models (LMMs) on mobile devices has gained significant attention due to the benefits of enhanced privacy, stability, and personalization. However, the hardware constraints of mobile devices necessitate the use of models with fewer parameters and model compression techniques like quantization. Currently, there is limited understanding of quantization's impact on various task performances, including LLM tasks, LMM tasks, and, critically, trust and safety. There is a lack of adequate tools for systematically testing these models on mobile devices. To address these gaps, we introduce MobileAIBench, a comprehensive benchmarking framework for evaluating mobile-optimized LLMs and LMMs. MobileAIBench assesses models across different sizes, quantization levels, and tasks, measuring latency and resource consumption on real devices. Our two-part open-source framework includes a library for running evaluations on desktops and an iOS app for on-device latency and hardware utilization measurements. Our thorough analysis aims to accelerate mobile AI research and deployment by providing insights into the performance and feasibility of deploying LLMs and LMMs on mobile platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.16355v3">RedCoast: A Lightweight Tool to Automate Distributed Training of LLMs on Any GPU/TPUs</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 RedCoast (Redco) has been released under Apache License 2.0 at https://github.com/tanyuqian/redco
    </div>
    <details class="paper-abstract">
      The recent progress of AI can be largely attributed to large language models (LLMs). However, their escalating memory requirements introduce challenges for machine learning (ML) researchers and engineers. Addressing this requires developers to partition a large model to distribute it across multiple GPUs or TPUs. This necessitates considerable coding and intricate configuration efforts with existing model parallel tools, such as Megatron-LM, DeepSpeed, and Alpa. These tools require users' expertise in machine learning systems (MLSys), creating a bottleneck in LLM development, particularly for developers without MLSys background. In this work, we present RedCoast (Redco), a lightweight and user-friendly tool crafted to automate distributed training and inference for LLMs, as well as to simplify ML pipeline development. The design of Redco emphasizes two key aspects. Firstly, to automate model parallelism, our study identifies two straightforward rules to generate tensor parallel strategies for any given LLM. Integrating these rules into Redco facilitates effortless distributed LLM training and inference, eliminating the need of additional coding or complex configurations. We demonstrate the effectiveness by applying Redco on a set of LLM architectures, such as GPT-J, LLaMA, T5, and OPT, up to the size of 66B. Secondly, we propose a mechanism that allows for the customization of diverse ML pipelines through the definition of merely three functions, avoiding redundant and formulaic code like multi-host related processing. This mechanism proves adaptable across a spectrum of ML algorithms, from foundational language modeling to complex algorithms like meta-learning and reinforcement learning. As a result, Redco implementations exhibit significantly fewer lines of code compared to their official counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09459v1">Ad Auctions for LLMs via Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      In the field of computational advertising, the integration of ads into the outputs of large language models (LLMs) presents an opportunity to support these services without compromising content integrity. This paper introduces novel auction mechanisms for ad allocation and pricing within the textual outputs of LLMs, leveraging retrieval-augmented generation (RAG). We propose a segment auction where an ad is probabilistically retrieved for each discourse segment (paragraph, section, or entire output) according to its bid and relevance, following the RAG framework, and priced according to competing bids. We show that our auction maximizes logarithmic social welfare, a new notion of welfare that balances allocation efficiency and fairness, and we characterize the associated incentive-compatible pricing rule. These results are extended to multi-ad allocation per segment. An empirical evaluation validates the feasibility and effectiveness of our approach over several ad auction scenarios, and exhibits inherent tradeoffs in metrics as we allow the LLM more flexibility to allocate ads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08657v1">Mistral-C2F: Coarse to Fine Actor for Analytical and Reasoning Enhancement in RLHF and Effective-Merged LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Despite the advances in Large Language Models (LLMs), exemplified by models like GPT-4 and Claude, smaller-scale LLMs such as Llama and Mistral often struggle with generating in-depth and coherent dialogues. This paper presents a novel two-step Coarse-to-Fine Actor model to address the inherent limitations in conversational and analytical capabilities of small-sized LLMs. Our approach begins with the Policy-based Coarse Actor, employing a technique we term "Continuous Maximization". The Coarse Actor establishes an enhanced, knowledge-rich pool adept at aligning with human preference styles in analysis and reasoning. Through the RLHF process, it employs Continuous Maximization, a strategy that dynamically and adaptively extends the output length limit, enabling the generation of more detailed and analytical content. Subsequently, the Fine Actor refines this analytical content, addressing the generation of excessively redundant information from the Coarse Actor. We introduce a "Knowledge Residue Merger" approach, refining the content from the Coarse Actor and merging it with an existing Instruction model to improve quality, correctness, and reduce redundancies. We applied our methodology to the popular Mistral model, creating Mistral-C2F, which has demonstrated exceptional performance across 11 general language tasks and the MT-Bench Dialogue task, outperforming similar-scale models and even larger models with 13B and 30B parameters. Our model has significantly improved conversational and analytical reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08607v1">Reversing the Forget-Retain Objectives: An Efficient LLM Unlearning Framework from Logit Difference</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 21 pages, 11 figures
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) demonstrate extensive capability in learning from documents, LLM unlearning becomes an increasingly important research area to address concerns of LLMs in terms of privacy, copyright, etc. A conventional LLM unlearning task typically involves two goals: (1) The target LLM should forget the knowledge in the specified forget documents, and (2) it should retain the other knowledge that the LLM possesses, for which we assume access to a small number of retain documents. To achieve both goals, a mainstream class of LLM unlearning methods introduces an optimization framework with a combination of two objectives - maximizing the prediction loss on the forget documents while minimizing that on the retain documents, which suffers from two challenges, degenerated output and catastrophic forgetting. In this paper, we propose a novel unlearning framework called Unlearning from Logit Difference (ULD), which introduces an assistant LLM that aims to achieve the opposite of the unlearning goals: remembering the forget documents and forgetting the retain knowledge. ULD then derives the unlearned LLM by computing the logit difference between the target and the assistant LLMs. We show that such reversed objectives would naturally resolve both aforementioned challenges while significantly improving the training efficiency. Extensive experiments demonstrate that our method efficiently achieves the intended forgetting while preserving the LLM's overall capabilities, reducing training time by more than threefold. Notably, our method loses 0% of model utility on the ToFU benchmark, whereas baseline methods may sacrifice 17% of utility on average to achieve comparable forget quality. Our code will be publicly available at https://github.com/UCSB-NLP-Chang/ULD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08582v1">Exploring Fact Memorization and Style Imitation in LLMs Using QLoRA: An Experimental Study and Quality Assessment Methods</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 16 pages, 5 tables
    </div>
    <details class="paper-abstract">
      There are various methods for adapting LLMs to different domains. The most common methods are prompting, finetuning, and RAG. In this work, we explore the possibility of adapting a model using one of the PEFT methods - QLoRA. The experiment aims to simulate human responses based on their interviews. The simulation quality is assessed by comparing the quality of the style and the quality of the generated facts.
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08477v1">Improving LLMs for Recommendation with Out-Of-Vocabulary Tokens</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Characterizing users and items through vector representations is crucial for various tasks in recommender systems. Recent approaches attempt to apply Large Language Models (LLMs) in recommendation through a question and answer format, where real users and items (e.g., Item No.2024) are represented with in-vocabulary tokens (e.g., "item", "20", "24"). However, since LLMs are typically pretrained on natural language tasks, these in-vocabulary tokens lack the expressive power for distinctive users and items, thereby weakening the recommendation ability even after fine-tuning on recommendation tasks. In this paper, we explore how to effectively tokenize users and items in LLM-based recommender systems. We emphasize the role of out-of-vocabulary (OOV) tokens in addition to the in-vocabulary ones and claim the memorization of OOV tokens that capture correlations of users/items as well as diversity of OOV tokens. By clustering the learned representations from historical user-item interactions, we make the representations of user/item combinations share the same OOV tokens if they have similar properties. Furthermore, integrating these OOV tokens into the LLM's vocabulary allows for better distinction between users and items and enhanced capture of user-item relationships during fine-tuning on downstream tasks. Our proposed framework outperforms existing state-of-the-art methods across various downstream recommendation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08386v1">Banal Deception Human-AI Ecosystems: A Study of People's Perceptions of LLM-generated Deceptive Behaviour</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can provide users with false, inaccurate, or misleading information, and we consider the output of this type of information as what Natale (2021) calls `banal' deceptive behaviour. Here, we investigate peoples' perceptions of ChatGPT-generated deceptive behaviour and how this affects peoples' own behaviour and trust. To do this, we use a mixed-methods approach comprising of (i) an online survey with 220 participants and (ii) semi-structured interviews with 12 participants. Our results show that (i) the most common types of deceptive information encountered were over-simplifications and outdated information; (ii) humans' perceptions of trust and `worthiness' of talking to ChatGPT are impacted by `banal' deceptive behaviour; (iii) the perceived responsibility for deception is influenced by education level and the frequency of deceptive information; and (iv) users become more cautious after encountering deceptive information, but they come to trust the technology more when they identify advantages of using it. Our findings contribute to the understanding of human-AI interaction dynamics in the context of \textit{Deceptive AI Ecosystems}, and highlight the importance of user-centric approaches to mitigating the potential harms of deceptive AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04978v2">An Enhanced Prompt-Based LLM Reasoning Scheme via Knowledge Graph-Integrated Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate exceptional performance in a multitude of Natural Language Processing (NLP) tasks, they encounter challenges in practical applications, including issues with hallucinations, inadequate knowledge updating, and limited transparency in the reasoning process. To overcome these limitations, this study innovatively proposes a collaborative training-free reasoning scheme involving tight cooperation between Knowledge Graph (KG) and LLMs. This scheme first involves using LLMs to iteratively explore KG, selectively retrieving a task-relevant knowledge subgraph to support reasoning. The LLMs are then guided to further combine inherent implicit knowledge to reason on the subgraph while explicitly elucidating the reasoning process. Through such a cooperative approach, our scheme achieves more reliable knowledge-based reasoning and facilitates the tracing of the reasoning results. Experimental results show that our scheme significantly progressed across multiple datasets, notably achieving over a 10% improvement on the QALD10 dataset compared to the best baseline and the fine-tuned state-of-the-art (SOTA) work. Building on this success, this study hopes to offer a valuable reference for future research in the fusion of KG and LLMs, thereby enhancing LLMs' proficiency in solving complex issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08334v1">ProTrain: Efficient LLM Training via Memory-Aware Techniques</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      It is extremely memory-hungry to train Large Language Models (LLM). To solve this problem, existing work exploits the combination of CPU and GPU for the training process, such as ZeRO-Offload. Such a technique largely democratizes billion-scale model training, making it possible to train with few consumer graphics cards. However, based on our observation, existing frameworks often provide coarse-grained memory management and require experienced experts in configuration tuning, leading to suboptimal hardware utilization and performance. This paper proposes ProTrain, a novel training system that intelligently balances memory usage and performance by coordinating memory, computation, and IO. ProTrain achieves adaptive memory management through Chunk-Based Model State Management and Block-Wise Activation Management, guided by a Memory-Aware Runtime Profiler without user intervention. ProTrain does not change the training algorithm and thus does not compromise accuracy. Experiments show that ProTrain improves training throughput by 1.43$\times$ to 2.71$\times$ compared to the SOTA training systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08305v1">Large Language Model(LLM) assisted End-to-End Network Health Management based on Multi-Scale Semanticization</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Network device and system health management is the foundation of modern network operations and maintenance. Traditional health management methods, relying on expert identification or simple rule-based algorithms, struggle to cope with the dynamic heterogeneous networks (DHNs) environment. Moreover, current state-of-the-art distributed anomaly detection methods, which utilize specific machine learning techniques, lack multi-scale adaptivity for heterogeneous device information, resulting in unsatisfactory diagnostic accuracy for DHNs. In this paper, we develop an LLM-assisted end-to-end intelligent network health management framework. The framework first proposes a Multi-Scale Semanticized Anomaly Detection Model (MSADM), incorporating semantic rule trees with an attention mechanism to address the multi-scale anomaly detection problem in DHNs. Secondly, a chain-of-thought-based large language model is embedded in downstream to adaptively analyze the fault detection results and produce an analysis report with detailed fault information and optimization strategies. Experimental results show that the accuracy of our proposed MSADM for heterogeneous network entity anomaly detection is as high as 91.31\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08337v2">LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 20 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Traffic congestion in metropolitan areas presents a formidable challenge with far-reaching economic, environmental, and societal ramifications. Therefore, effective congestion management is imperative, with traffic signal control (TSC) systems being pivotal in this endeavor. Conventional TSC systems, designed upon rule-based algorithms or reinforcement learning (RL), frequently exhibit deficiencies in managing the complexities and variabilities of urban traffic flows, constrained by their limited capacity for adaptation to unfamiliar scenarios. In response to these limitations, this work introduces an innovative approach that integrates Large Language Models (LLMs) into TSC, harnessing their advanced reasoning and decision-making faculties. Specifically, a hybrid framework that augments LLMs with a suite of perception and decision-making tools is proposed, facilitating the interrogation of both the static and dynamic traffic information. This design places the LLM at the center of the decision-making process, combining external traffic data with established TSC methods. Moreover, a simulation platform is developed to corroborate the efficacy of the proposed framework. The findings from our simulations attest to the system's adeptness in adjusting to a multiplicity of traffic environments without the need for additional training. Notably, in cases of Sensor Outage (SO), our approach surpasses conventional RL-based systems by reducing the average waiting time by $20.4\%$. This research signifies a notable advance in TSC strategies and paves the way for the integration of LLMs into real-world, dynamic scenarios, highlighting their potential to revolutionize traffic management. The related code is available at https://github.com/Traffic-Alpha/LLM-Assisted-Light.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.09605v3">Penetrative AI: Making LLMs Comprehend the Physical World</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 ACL Findings 2024
    </div>
    <details class="paper-abstract">
      Recent developments in Large Language Models (LLMs) have demonstrated their remarkable capabilities across a range of tasks. Questions, however, persist about the nature of LLMs and their potential to integrate common-sense human knowledge when performing tasks involving information about the real physical world. This paper delves into these questions by exploring how LLMs can be extended to interact with and reason about the physical world through IoT sensors and actuators, a concept that we term "Penetrative AI". The paper explores such an extension at two levels of LLMs' ability to penetrate into the physical world via the processing of sensory signals. Our preliminary findings indicate that LLMs, with ChatGPT being the representative example in our exploration, have considerable and unique proficiency in employing the embedded world knowledge for interpreting IoT sensor data and reasoning over them about tasks in the physical realm. Not only this opens up new applications for LLMs beyond traditional text-based tasks, but also enables new ways of incorporating human knowledge in cyber-physical systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08184v1">MobileAgentBench: An Efficient and User-Friendly Benchmark for Mobile LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based mobile agents are increasingly popular due to their capability to interact directly with mobile phone Graphic User Interfaces (GUIs) and their potential to autonomously manage daily tasks. Despite their promising prospects in both academic and industrial sectors, little research has focused on benchmarking the performance of existing mobile agents, due to the inexhaustible states of apps and the vague definition of feasible action sequences. To address this challenge, we propose an efficient and user-friendly benchmark, MobileAgentBench, designed to alleviate the burden of extensive manual testing. We initially define 100 tasks across 10 open-source apps, categorized by multiple levels of difficulty. Subsequently, we evaluate several existing mobile agents, including AppAgent and MobileAgent, to thoroughly and systematically compare their performance. All materials are accessible on our project webpage: https://MobileAgentBench.github.io, contributing to the advancement of both academic and industrial fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08120v1">Interlinking User Stories and GUI Prototyping: A Semi-Automatic LLM-based Approach</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Interactive systems are omnipresent today and the need to create graphical user interfaces (GUIs) is just as ubiquitous. For the elicitation and validation of requirements, GUI prototyping is a well-known and effective technique, typically employed after gathering initial user requirements represented in natural language (NL) (e.g., in the form of user stories). Unfortunately, GUI prototyping often requires extensive resources, resulting in a costly and time-consuming process. Despite various easy-to-use prototyping tools in practice, there is often a lack of adequate resources for developing GUI prototypes based on given user requirements. In this work, we present a novel Large Language Model (LLM)-based approach providing assistance for validating the implementation of functional NL-based requirements in a GUI prototype embedded in a prototyping tool. In particular, our approach aims to detect functional user stories that are not implemented in a GUI prototype and provides recommendations for suitable GUI components directly implementing the requirements. We collected requirements for existing GUIs in the form of user stories and evaluated our proposed validation and recommendation approach with this dataset. The obtained results are promising for user story validation and we demonstrate feasibility for the GUI component recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08112v1">Codecfake: An Initial Dataset for Detecting LLM-based Deepfake Audio</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 Accepted by INTERSPEECH 2024. arXiv admin note: substantial text overlap with arXiv:2405.04880
    </div>
    <details class="paper-abstract">
      With the proliferation of Large Language Model (LLM) based deepfake audio, there is an urgent need for effective detection methods. Previous deepfake audio generation methods typically involve a multi-step generation process, with the final step using a vocoder to predict the waveform from handcrafted features. However, LLM-based audio is directly generated from discrete neural codecs in an end-to-end generation process, skipping the final step of vocoder processing. This poses a significant challenge for current audio deepfake detection (ADD) models based on vocoder artifacts. To effectively detect LLM-based deepfake audio, we focus on the core of the generation process, the conversion from neural codec to waveform. We propose Codecfake dataset, which is generated by seven representative neural codec methods. Experiment results show that codec-trained ADD models exhibit a 41.406% reduction in average equal error rate compared to vocoder-trained ADD models on the Codecfake test set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06590v2">Are LLMs classical or nonmonotonic reasoners? Lessons from generics</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 Accepted at ACL 2024 (main)
    </div>
    <details class="paper-abstract">
      Recent scholarship on reasoning in LLMs has supplied evidence of impressive performance and flexible adaptation to machine generated or human feedback. Nonmonotonic reasoning, crucial to human cognition for navigating the real world, remains a challenging, yet understudied task. In this work, we study nonmonotonic reasoning capabilities of seven state-of-the-art LLMs in one abstract and one commonsense reasoning task featuring generics, such as 'Birds fly', and exceptions, 'Penguins don't fly' (see Fig. 1). While LLMs exhibit reasoning patterns in accordance with human nonmonotonic reasoning abilities, they fail to maintain stable beliefs on truth conditions of generics at the addition of supporting examples ('Owls fly') or unrelated information ('Lions have manes'). Our findings highlight pitfalls in attributing human reasoning behaviours to LLMs, as well as assessing general capabilities, while consistent reasoning remains elusive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.12519v3">DPIC: Decoupling Prompt and Intrinsic Characteristics for LLM Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have the potential to generate texts that pose risks of misuse, such as plagiarism, planting fake reviews on e-commerce platforms, or creating inflammatory false tweets. Consequently, detecting whether a text is generated by LLMs has become increasingly important. Existing high-quality detection methods usually require access to the interior of the model to extract the intrinsic characteristics. However, since we do not have access to the interior of the black-box model, we must resort to surrogate models, which impacts detection quality. In order to achieve high-quality detection of black-box models, we would like to extract deep intrinsic characteristics of the black-box model generated texts. We view the generation process as a coupled process of prompt and intrinsic characteristics of the generative model. Based on this insight, we propose to decouple prompt and intrinsic characteristics (DPIC) for LLM-generated text detection method. Specifically, given a candidate text, DPIC employs an auxiliary LLM to reconstruct the prompt corresponding to the candidate text, then uses the prompt to regenerate text by the auxiliary LLM, which makes the candidate text and the regenerated text align with their prompts, respectively. Then, the similarity between the candidate text and the regenerated text is used as a detection feature, thus eliminating the prompt in the detection process, which allows the detector to focus on the intrinsic characteristics of the generative model. Compared to the baselines, DPIC has achieved an average improvement of 6.76\% and 2.91\% in detecting texts from different domains generated by GPT4 and Claude3, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07954v1">Dataset and Lessons Learned from the 2024 SaTML LLM Capture-the-Flag Competition</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Large language model systems face important security risks from maliciously crafted messages that aim to overwrite the system's original instructions or leak private data. To study this problem, we organized a capture-the-flag competition at IEEE SaTML 2024, where the flag is a secret string in the LLM system prompt. The competition was organized in two phases. In the first phase, teams developed defenses to prevent the model from leaking the secret. During the second phase, teams were challenged to extract the secrets hidden for defenses proposed by the other teams. This report summarizes the main insights from the competition. Notably, we found that all defenses were bypassed at least once, highlighting the difficulty of designing a successful defense and the necessity for additional research to protect LLM systems. To foster future research in this direction, we compiled a dataset with over 137k multi-turn attack chats and open-sourced the platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02567v2">Eliciting Better Multilingual Structured Reasoning from LLMs through Code</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      The development of large language models (LLM) has shown progress on reasoning, though studies have largely considered either English or simple reasoning tasks. To address this, we introduce a multilingual structured reasoning and explanation dataset, termed xSTREET, that covers four tasks across six languages. xSTREET exposes a gap in base LLM performance between English and non-English reasoning tasks. We then propose two methods to remedy this gap, building on the insight that LLMs trained on code are better reasoners. First, at training time, we augment a code dataset with multilingual comments using machine translation while keeping program code as-is. Second, at inference time, we bridge the gap between training and inference by employing a prompt structure that incorporates step-by-step code primitives to derive new facts and find a solution. Our methods show improved multilingual performance on xSTREET, most notably on the scientific commonsense reasoning subtask. Furthermore, the models show no regression on non-reasoning tasks, thus demonstrating our techniques maintain general-purpose abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07944v1">DLLens: Testing Deep Learning Libraries via LLM-aided Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Testing is a major approach to ensuring the quality of deep learning (DL) libraries. Existing testing techniques commonly adopt differential testing to relieve the need for test oracle construction. However, these techniques are limited in finding implementations that offer the same functionality and generating diverse test inputs for differential testing. This paper introduces DLLens, a novel differential testing technique for DL library testing. Our insight is that APIs in different DL libraries are commonly designed to accomplish various computations for the same set of published DL algorithms. Although the mapping of these APIs is not often one-to-one, we observe that their computations can be mutually simulated after proper composition and adaptation. The use of these simulation counterparts facilitates differential testing for the detection of functional DL library bugs. Leveraging the insight, we propose DLLens as a novel mechanism that utilizes a large language model (LLM) to synthesize valid counterparts of DL library APIs. To generate diverse test inputs, DLLens incorporates a static analysis method aided by LLM to extract path constraints from all execution paths in each API and its counterpart's implementations. These path constraints are then used to guide the generation of diverse test inputs. We evaluate DLLens on two popular DL libraries, TensorFlow and PyTorch. Our evaluation shows that DLLens can synthesize counterparts for more than twice as many APIs found by state-of-the-art techniques on these libraries. Moreover, DLLens can extract 26.7% more constraints and detect 2.5 times as many bugs as state-of-the-art techniques. DLLens has successfully found 56 bugs in recent TensorFlow and PyTorch libraries. Among them, 41 are previously unknown, 39 of which have been confirmed by developers after reporting, and 19 of those confirmed bugs have been fixed by developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06156v2">Stronger, Cheaper and Demonstration-Free Log Parsing with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Log parsing, the process of converting raw log messages into structured formats, is an important initial step for automated analysis of logs of large-scale software systems. Traditional log parsers often rely on heuristics or handcrafted features, which may not generalize well across diverse log sources or require extensive model tuning. Recently, some log parsers have utilized powerful generative capabilities of large language models (LLMs). However, they heavily rely on demonstration examples, resulting in substantial overhead in LLM invocations. To address these issues, we propose LogBatcher, a cost-effective LLM-based log parser that requires no training process or labeled data. To leverage latent characteristics of log data and reduce the overhead, we divide logs into several partitions through clustering. Then we perform a cache matching process to match logs with previously parsed log templates. Finally, we provide LLMs with better prompt context specialized for log parsing by batching a group of logs from each partition. We have conducted experiments on 16 public log datasets and the results show that LogBatcher is effective and efficient for log parsing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.04319v3">Know Your Needs Better: Towards Structured Understanding of Marketer Demands with Analogical Reasoning Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 Accepted by KDD 2024
    </div>
    <details class="paper-abstract">
      In this paper, we explore a new way for user targeting, where non-expert marketers could select their target users solely given demands in natural language form. The key to this issue is how to transform natural languages into practical structured logical languages, i.e., the structured understanding of marketer demands. In practical scenarios, the demands of non-expert marketers are often abstract and diverse. Considering the impressive natural language processing ability of large language models (LLMs), we try to leverage LLMs to solve this issue. To stimulate the LLMs' reasoning ability, the chain-of-thought (CoT) prompting method is widely used, but existing methods still have some limitations in our scenario: (1) Previous methods either use simple "Let's think step by step" spells or provide fixed examples in demonstrations without considering compatibility between prompts and concrete questions, making LLMs ineffective when the marketers' demands are abstract and diverse. (2) Previous methods are often implemented in closed-source models or excessively large models, which is not suitable in industrial practical scenarios. Based on these, we propose ARALLM (i.e., Analogical Reasoning Augmented Large Language Models) consisting of two modules: Analogical Reasoning based Prompting and Reasoning-Augmented Multi-Task Model Distillation. Part of our data and code can be found at https://github.com/alipay/Analogic-Reasoning-Augmented-Large-Language-Model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02060v3">Long-context LLMs Struggle with Long In-context Learning</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant strides in handling long sequences. Some models like Gemini could even to be capable of dealing with millions of tokens. However, their performance evaluation has largely been confined to metrics like perplexity and synthetic tasks, which may not fully capture their true abilities in more challenging, real-world scenarios. We introduce a benchmark (LongICLBench) for long in-context learning in extreme-label classification using six datasets with 28 to 174 classes and input lengths from 2K to 50K tokens. Our benchmark requires LLMs to comprehend the entire input to recognize the massive label spaces to make correct predictions. We evaluate on 15 long-context LLMs and find that they perform well on less challenging classification tasks with smaller label space and shorter demonstrations. However, they struggle with more challenging task like Discovery with 174 labels, suggesting a gap in their ability to process long, context-rich sequences. Further analysis reveals a bias towards labels presented later in the sequence and a need for improved reasoning over multiple pieces of information. Our study reveals that long context understanding and reasoning is still a challenging task for the existing LLMs. We believe LongICLBench could serve as a more realistic evaluation for the future long-context LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01817v3">LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      There is considerable confusion about the role of Large Language Models (LLMs) in planning and reasoning tasks. On one side are over-optimistic claims that LLMs can indeed do these tasks with just the right prompting or self-verification strategies. On the other side are perhaps over-pessimistic claims that all that LLMs are good for in planning/reasoning tasks are as mere translators of the problem specification from one syntactic format to another, and ship the problem off to external symbolic solvers. In this position paper, we take the view that both these extremes are misguided. We argue that auto-regressive LLMs cannot, by themselves, do planning or self-verification (which is after all a form of reasoning), and shed some light on the reasons for misunderstandings in the literature. We will also argue that LLMs should be viewed as universal approximate knowledge sources that have much more meaningful roles to play in planning/reasoning tasks beyond simple front-end/back-end format translators. We present a vision of {\bf LLM-Modulo Frameworks} that combine the strengths of LLMs with external model-based verifiers in a tighter bi-directional interaction regime. We will show how the models driving the external verifiers themselves can be acquired with the help of LLMs. We will also argue that rather than simply pipelining LLMs and symbolic components, this LLM-Modulo Framework provides a better neuro-symbolic approach that offers tighter integration between LLMs and symbolic components, and allows extending the scope of model-based planning/reasoning regimes towards more flexible knowledge, problem and preference specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.14348v3">Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 19 Pages, 5 Figures, 8 Tables. Accepted by ACL 2024
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.11457v2">When Do LLMs Need Retrieval Augmentation? Mitigating LLMs' Overconfidence Helps Retrieval Augmentation</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been found to have difficulty knowing they do not possess certain knowledge and tend to provide specious answers in such cases. Retrieval Augmentation (RA) has been extensively studied to mitigate LLMs' hallucinations. However, due to the extra overhead and unassured quality of retrieval, it may not be optimal to conduct RA all the time. A straightforward idea is to only conduct retrieval when LLMs are uncertain about a question. This motivates us to enhance the LLMs' ability to perceive their knowledge boundaries to help RA. In this paper, we first quantitatively measure LLMs' such ability and confirm their overconfidence. Then, we study how LLMs' certainty about a question correlates with their dependence on external retrieved information. We propose several methods to enhance LLMs' perception of knowledge boundaries and show that they are effective in reducing overconfidence. Additionally, equipped with these methods, LLMs can achieve comparable or even better performance of RA with much fewer retrieval calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.01817v3">LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      There is considerable confusion about the role of Large Language Models (LLMs) in planning and reasoning tasks. On one side are over-optimistic claims that LLMs can indeed do these tasks with just the right prompting or self-verification strategies. On the other side are perhaps over-pessimistic claims that all that LLMs are good for in planning/reasoning tasks are as mere translators of the problem specification from one syntactic format to another, and ship the problem off to external symbolic solvers. In this position paper, we take the view that both these extremes are misguided. We argue that auto-regressive LLMs cannot, by themselves, do planning or self-verification (which is after all a form of reasoning), and shed some light on the reasons for misunderstandings in the literature. We will also argue that LLMs should be viewed as universal approximate knowledge sources that have much more meaningful roles to play in planning/reasoning tasks beyond simple front-end/back-end format translators. We present a vision of {\bf LLM-Modulo Frameworks} that combine the strengths of LLMs with external model-based verifiers in a tighter bi-directional interaction regime. We will show how the models driving the external verifiers themselves can be acquired with the help of LLMs. We will also argue that rather than simply pipelining LLMs and symbolic components, this LLM-Modulo Framework provides a better neuro-symbolic approach that offers tighter integration between LLMs and symbolic components, and allows extending the scope of model-based planning/reasoning regimes towards more flexible knowledge, problem and preference specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06950v1">A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation</a></div>
    <div class="paper-meta">
      📅 2024-06-11
      | 💬 26 pages, 18 figures
    </div>
    <details class="paper-abstract">
      This paper focuses on the task of hallucination detection, which aims to determine the truthfulness of LLM-generated statements. To address this problem, a popular class of methods utilize the LLM's self-consistencies in its beliefs in a set of logically related augmented statements generated by the LLM, which does not require external knowledge databases and can work with both white-box and black-box LLMs. However, in many existing approaches, the augmented statements tend to be very monotone and unstructured, which makes it difficult to integrate meaningful information from the LLM beliefs in these statements. Also, many methods work with the binarized version of the LLM's belief, instead of the continuous version, which significantly loses information. To overcome these limitations, in this paper, we propose Belief Tree Propagation (BTProp), a probabilistic framework for LLM hallucination detection. BTProp introduces a belief tree of logically related statements by recursively decomposing a parent statement into child statements with three decomposition strategies, and builds a hidden Markov tree model to integrate the LLM's belief scores in these statements in a principled way. Experiment results show that our method improves baselines by 3%-9% (evaluated by AUROC and AUC-PR) on multiple hallucination detection benchmarks. Code is available at https://github.com/UCSB-NLP-Chang/BTProp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06918v1">Towards more realistic evaluation of LLM-based code generation: an experimental study and beyond</a></div>
    <div class="paper-meta">
      📅 2024-06-11
    </div>
    <details class="paper-abstract">
      To evaluate the code generation capabilities of Large Language Models (LLMs) in complex real-world software development scenarios, many evaluation approaches have been developed. They typically leverage contextual code from the latest version of a project to facilitate LLMs in accurately generating the desired function. However, such evaluation approaches fail to consider the dynamic evolution of software projects over time, which we refer to as evolving-ignored situation, leading to issues of future context leakage and useful context missing. This in turn results in inaccurate evaluation of LLMs' performance. In this paper, we conduct an empirical study to deeply understand LLMs' code generation performance within settings that reflect the evolving nature of software development. To achieve this, we first construct an evolving-aware repository-level code generation dataset, namely HumanEvo, equipped with an automated execution-based evaluation tool. Second, we manually categorize HumanEvo according to dependency levels to more comprehensively analyze the model's performance in generating functions with different dependency levels. Third, we conduct extensive experiments on HumanEvo with seven representative and diverse LLMs to verify the effectiveness of the proposed benchmark. We obtain many important findings through our experimental study. For example, we find that previous evolving-ignored evaluation approaches lead to inflated performance of the LLMs, ranging from 10.0% to 61.1%. Based on the findings, we give actionable suggestions on more realistic evaluation of LLMs on code generation. We also build a shared evolving-aware code generation toolbox to facilitate future research. Replication package including source code, datasets and appendix is available at https://github.com/DeepSoftwareAnalytics/EvoEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05955v2">Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters</a></div>
    <div class="paper-meta">
      📅 2024-06-11
    </div>
    <details class="paper-abstract">
      Exploiting activation sparsity is a promising approach to significantly accelerating the inference process of large language models (LLMs) without compromising performance. However, activation sparsity is determined by activation functions, and commonly used ones like SwiGLU and GeGLU exhibit limited sparsity. Simply replacing these functions with ReLU fails to achieve sufficient sparsity. Moreover, inadequate training data can further increase the risk of performance degradation. To address these challenges, we propose a novel dReLU function, which is designed to improve LLM activation sparsity, along with a high-quality training data mixture ratio to facilitate effective sparsification. Additionally, we leverage sparse activation patterns within the Feed-Forward Network (FFN) experts of Mixture-of-Experts (MoE) models to further boost efficiency. By applying our neuron sparsification method to the Mistral and Mixtral models, only 2.5 billion and 4.3 billion parameters are activated per inference iteration, respectively, while achieving even more powerful model performance. Evaluation results demonstrate that this sparsity achieves a 2-5x decoding speedup. Remarkably, on mobile phones, our TurboSparse-Mixtral-47B achieves an inference speed of 11 tokens per second. Our models are available at \url{https://huggingface.co/PowerInfer}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06864v1">Validating LLM-Generated Programs with Metamorphic Prompt Testing</a></div>
    <div class="paper-meta">
      📅 2024-06-11
    </div>
    <details class="paper-abstract">
      The latest paradigm shift in software development brings in the innovation and automation afforded by Large Language Models (LLMs), showcased by Generative Pre-trained Transformer (GPT), which has shown remarkable capacity to generate code autonomously, significantly reducing the manual effort required for various programming tasks. Although, the potential benefits of LLM-generated code are vast, most notably in efficiency and rapid prototyping, as LLMs become increasingly integrated into the software development lifecycle and hence the supply chain, complex and multifaceted challenges arise as the code generated from these language models carry profound questions on quality and correctness. Research is required to comprehensively explore these critical concerns surrounding LLM-generated code. In this paper, we propose a novel solution called metamorphic prompt testing to address these challenges. Our intuitive observation is that intrinsic consistency always exists among correct code pieces but may not exist among flawed code pieces, so we can detect flaws in the code by detecting inconsistencies. Therefore, we can vary a given prompt to multiple prompts with paraphrasing, and to ask the LLM to acquire multiple versions of generated code, so that we can validate whether the semantic relations still hold in the acquired code through cross-validation. Our evaluation on HumanEval shows that metamorphic prompt testing is able to detect 75 percent of the erroneous programs generated by GPT-4, with a false positive rate of 8.6 percent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06863v1">Ollabench: Evaluating LLMs' Reasoning for Human-centric Interdependent Cybersecurity</a></div>
    <div class="paper-meta">
      📅 2024-06-11
      | 💬 12 pages, 7 figures, 2 tables The final conference/journal version may have significantly more content updates
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the potential to enhance Agent-Based Modeling by better representing complex interdependent cybersecurity systems, improving cybersecurity threat modeling and risk management. However, evaluating LLMs in this context is crucial for legal compliance and effective application development. Existing LLM evaluation frameworks often overlook the human factor and cognitive computing capabilities essential for interdependent cybersecurity. To address this gap, I propose OllaBench, a novel evaluation framework that assesses LLMs' accuracy, wastefulness, and consistency in answering scenario-based information security compliance and non-compliance questions. OllaBench is built on a foundation of 24 cognitive behavioral theories and empirical evidence from 38 peer-reviewed papers. OllaBench was used to evaluate 21 LLMs, including both open-weight and commercial models from OpenAI, Anthropic, Google, Microsoft, Meta and so on. The results reveal that while commercial LLMs have the highest overall accuracy scores, there is significant room for improvement. Smaller low-resolution open-weight LLMs are not far behind in performance, and there are significant differences in token efficiency and consistency among the evaluated models. OllaBench provides a user-friendly interface and supports a wide range of LLM platforms, making it a valuable tool for researchers and solution developers in the field of human-centric interdependent cybersecurity and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.02107v4">Instances Need More Care: Rewriting Prompts for Instances with LLMs in the Loop Yields Better Zero-Shot Performance</a></div>
    <div class="paper-meta">
      📅 2024-06-11
      | 💬 Accepted at ACL 2024 - Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized zero-shot task performance, mitigating the need for task-specific annotations while enhancing task generalizability. Despite its advancements, current methods using trigger phrases such as "Let's think step by step" remain limited. This study introduces PRomPTed, an approach that optimizes the zero-shot prompts for individual task instances following an innovative manner of "LLMs in the loop". Our comprehensive evaluation across 13 datasets and 10 task types based on GPT-4 reveals that PRomPTed significantly outperforms both the naive zero-shot approaches and a strong baseline (i.e., "Output Refinement") which refines the task output instead of the input prompt. Our experimental results also confirmed the generalization of this advantage to the relatively weaker GPT-3.5. Even more intriguingly, we found that leveraging GPT-3.5 to rewrite prompts for the stronger GPT-4 not only matches but occasionally exceeds the efficacy of using GPT-4 as the prompt rewriter. Our research thus presents a huge value in not only enhancing zero-shot LLM performance but also potentially enabling supervising LLMs with their weaker counterparts, a capability attracting much interest recently. Finally, our additional experiments confirm the generalization of the advantages to open-source LLMs such as Mistral 7B and Mixtral 8x7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15842v3">Knowledge Distillation of LLM for Automatic Scoring of Science Education Assessments</a></div>
    <div class="paper-meta">
      📅 2024-06-11
      | 💬 Accepted to AIED2024
    </div>
    <details class="paper-abstract">
      This study proposes a method for knowledge distillation (KD) of fine-tuned Large Language Models (LLMs) into smaller, more efficient, and accurate neural networks. We specifically target the challenge of deploying these models on resource-constrained devices. Our methodology involves training the smaller student model (Neural Network) using the prediction probabilities (as soft labels) of the LLM, which serves as a teacher model. This is achieved through a specialized loss function tailored to learn from the LLM's output probabilities, ensuring that the student model closely mimics the teacher's performance. To validate the performance of the KD approach, we utilized a large dataset, 7T, containing 6,684 student-written responses to science questions and three mathematical reasoning datasets with student-written responses graded by human experts. We compared accuracy with state-of-the-art (SOTA) distilled models, TinyBERT, and artificial neural network (ANN) models. Results have shown that the KD approach has 3% and 2% higher scoring accuracy than ANN and TinyBERT, respectively, and comparable accuracy to the teacher model. Furthermore, the student model size is 0.03M, 4,000 times smaller in parameters and x10 faster in inferencing than the teacher model and TinyBERT, respectively. The significance of this research lies in its potential to make advanced AI technologies accessible in typical educational settings, particularly for automatic scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07545v1">Open-LLM-Leaderboard: From Multi-choice to Open-style Questions for LLMs Evaluation, Benchmark, and Arena</a></div>
    <div class="paper-meta">
      📅 2024-06-11
      | 💬 Code and dataset are available at https://github.com/VILA-Lab/Open-LLM-Leaderboard
    </div>
    <details class="paper-abstract">
      Multiple-choice questions (MCQ) are frequently used to assess large language models (LLMs). Typically, an LLM is given a question and selects the answer deemed most probable after adjustments for factors like length. Unfortunately, LLMs may inherently favor certain answer choice IDs, such as A/B/C/D, due to inherent biases of priori unbalanced probabilities, influencing the prediction of answers based on these IDs. Previous research has introduced methods to reduce this ''selection bias'' by simply permutating options on a few test samples and applying to new ones. Another problem of MCQ is the lottery ticket choice by ''random guessing''. The LLM does not learn particular knowledge, but the option is guessed correctly. This situation is especially serious for those small-scale LLMs. To address them, a more thorough approach involves shifting from MCQ to open-style questions, which can fundamentally eliminate selection bias and random guessing issues. However, transitioning causes its own set of challenges in (1) identifying suitable open-style questions and (2) validating the correctness of LLM open-style responses against human-annotated ground-truths. This work aims to tackle these significant difficulties, and establish a new LLM evaluation benchmark through entirely open-style questions. Consequently, we introduce the Open-LLM-Leaderboard to track various LLMs' performance and reflect true capability of them, such as GPT-4o/4/3.5, Claude 3, Gemini, etc. Our code and dataset are available at https://github.com/VILA-Lab/Open-LLM-Leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07400v1">Guiding LLM Temporal Logic Generation with Explicit Separation of Data and Control</a></div>
    <div class="paper-meta">
      📅 2024-06-11
    </div>
    <details class="paper-abstract">
      Temporal logics are powerful tools that are widely used for the synthesis and verification of reactive systems. The recent progress on Large Language Models (LLMs) has the potential to make the process of writing such specifications more accessible. However, writing specifications in temporal logics remains challenging for all but the most expert users. A key question in using LLMs for temporal logic specification engineering is to understand what kind of guidance is most helpful to the LLM and the users to easily produce specifications. Looking specifically at the problem of reactive program synthesis, we explore the impact of providing an LLM with guidance on the separation of control and data--making explicit for the LLM what functionality is relevant for the specification, and treating the remaining functionality as an implementation detail for a series of pre-defined functions and predicates. We present a benchmark set and find that this separation of concerns improves specification generation. Our benchmark provides a test set against which to verify future work in LLM generation of temporal logic specifications.
    </details>
</div>
