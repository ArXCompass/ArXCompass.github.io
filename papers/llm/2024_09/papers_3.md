# llm - 2024_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18998v1">Controlled LLM-based Reasoning for Clinical Trial Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      Matching patients to clinical trials demands a systematic and reasoned interpretation of documents which require significant expert-level background knowledge, over a complex set of well-defined eligibility criteria. Moreover, this interpretation process needs to operate at scale, over vast knowledge bases of trials. In this paper, we propose a scalable method that extends the capabilities of LLMs in the direction of systematizing the reasoning over sets of medical eligibility criteria, evaluating it in the context of real-world cases. The proposed method overlays a Set-guided reasoning method for LLMs. The proposed framework is evaluated on TREC 2022 Clinical Trials, achieving results superior to the state-of-the-art: NDCG@10 of 0.693 and Precision@10 of 0.73.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12580v1">LLMs Can Check Their Own Results to Mitigate Hallucinations in Traffic Understanding Tasks</a></div>
    <div class="paper-meta">
      📅 2024-09-19
      | 💬 ICTSS 2024, 36th International Conference on Testing Software and Systems
    </div>
    <details class="paper-abstract">
      Today's Large Language Models (LLMs) have showcased exemplary capabilities, ranging from simple text generation to advanced image processing. Such models are currently being explored for in-vehicle services such as supporting perception tasks in Advanced Driver Assistance Systems (ADAS) or Autonomous Driving (AD) systems, given the LLMs' capabilities to process multi-modal data. However, LLMs often generate nonsensical or unfaithful information, known as ``hallucinations'': a notable issue that needs to be mitigated. In this paper, we systematically explore the adoption of SelfCheckGPT to spot hallucinations by three state-of-the-art LLMs (GPT-4o, LLaVA, and Llama3) when analysing visual automotive data from two sources: Waymo Open Dataset, from the US, and PREPER CITY dataset, from Sweden. Our results show that GPT-4o is better at generating faithful image captions than LLaVA, whereas the former demonstrated leniency in mislabeling non-hallucinated content as hallucinations compared to the latter. Furthermore, the analysis of the performance metrics revealed that the dataset type (Waymo or PREPER CITY) did not significantly affect the quality of the captions or the effectiveness of hallucination detection. However, the models showed better performance rates over images captured during daytime, compared to during dawn, dusk or night. Overall, the results show that SelfCheckGPT and its adaptation can be used to filter hallucinations in generated traffic-related image captions for state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12561v1">Human Interest or Conflict? Leveraging LLMs for Automated Framing Analysis in TV Shows</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      In the current media landscape, understanding the framing of information is crucial for critical consumption and informed decision making. Framing analysis is a valuable tool for identifying the underlying perspectives used to present information, and has been applied to a variety of media formats, including television programs. However, manual analysis of framing can be time-consuming and labor-intensive. This is where large language models (LLMs) can play a key role. In this paper, we propose a novel approach to use prompt-engineering to identify the framing of spoken content in television programs. Our findings indicate that prompt-engineering LLMs can be used as a support tool to identify frames, with agreement rates between human and machine reaching up to 43\%. As LLMs are still under development, we believe that our approach has the potential to be refined and further improved. The potential of this technology for interactive media applications is vast, including the development of support tools for journalists, educational resources for students of journalism learning about framing and related concepts, and interactive media experiences for audiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12538v1">PersonaFlow: Boosting Research Ideation with LLM-Simulated Expert Personas</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      Developing novel interdisciplinary research ideas often requires discussions and feedback from experts across different domains. However, obtaining timely inputs is challenging due to the scarce availability of domain experts. Recent advances in Large Language Model (LLM) research have suggested the feasibility of utilizing LLM-simulated expert personas to support research ideation. In this study, we introduce PersonaFlow, an LLM-based system using persona simulation to support the ideation stage of interdisciplinary scientific discovery. Our findings indicate that using multiple personas during ideation significantly enhances user-perceived quality of outcomes (e.g., relevance of critiques, creativity of research questions) without increasing cognitive load. We also found that users' persona customization interactions significantly improved their sense of control and recall of generated ideas. Based on the findings, we discuss highlighting ethical concerns, including potential over-reliance and cognitive biases, and suggest design implications for leveraging LLM-simulated expert personas to support research ideation when human expertise is inaccessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09380v2">The Midas Touch: Triggering the Capability of LLMs for RM-API Misuse Detection</a></div>
    <div class="paper-meta">
      📅 2024-09-19
      | 💬 Accepted by NDSS Symposium 2025. Please cite this paper as "Yi Yang, Jinghua Liu, Kai Chen, Miaoqian Lin. The Midas Touch: Triggering the Capability of LLMs for RM-API Misuse Detection. In the 32nd Annual Network and Distributed System Security Symposium (NDSS 2025)."
    </div>
    <details class="paper-abstract">
      In this paper, we propose an LLM-empowered RM-API misuse detection solution, ChatDetector, which fully automates LLMs for documentation understanding which helps RM-API constraints retrieval and RM-API misuse detection. To correctly retrieve the RM-API constraints, ChatDetector is inspired by the ReAct framework which is optimized based on Chain-of-Thought (CoT) to decompose the complex task into allocation APIs identification, RM-object (allocated/released by RM APIs) extraction and RM-APIs pairing (RM APIs usually exist in pairs). It first verifies the semantics of allocation APIs based on the retrieved RM sentences from API documentation through LLMs. Inspired by the LLMs' performance on various prompting methods,ChatDetector adopts a two-dimensional prompting approach for cross-validation. At the same time, an inconsistency-checking approach between the LLMs' output and the reasoning process is adopted for the allocation APIs confirmation with an off-the-shelf Natural Language Processing (NLP) tool. To accurately pair the RM-APIs, ChatDetector decomposes the task again and identifies the RM-object type first, with which it can then accurately pair the releasing APIs and further construct the RM-API constraints for misuse detection. With the diminished hallucinations, ChatDetector identifies 165 pairs of RM-APIs with a precision of 98.21% compared with the state-of-the-art API detectors. By employing a static detector CodeQL, we ethically report 115 security bugs on the applications integrating on six popular libraries to the developers, which may result in severe issues, such as Denial-of-Services (DoS) and memory corruption. Compared with the end-to-end benchmark method, the result shows that ChatDetector can retrieve at least 47% more RM sentences and 80.85% more RM-API constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09288v2">Generating API Parameter Security Rules with LLM for API Misuse Detection</a></div>
    <div class="paper-meta">
      📅 2024-09-19
      | 💬 Accepted by NDSS Symposium 2025. Please cite this paper as "Jinghua Liu, Yi Yang, Kai Chen, and Miaoqian Lin. Generating API Parameter Security Rules with LLM for API Misuse Detection. In the 32nd Annual Network and Distributed System Security Symposium (NDSS 2025)
    </div>
    <details class="paper-abstract">
      In this paper, we present a new framework, named GPTAid, for automatic APSRs generation by analyzing API source code with LLM and detecting API misuse caused by incorrect parameter use. To validate the correctness of the LLM-generated APSRs, we propose an execution feedback-checking approach based on the observation that security-critical API misuse is often caused by APSRs violations, and most of them result in runtime errors. Specifically, GPTAid first uses LLM to generate raw APSRs and the Right calling code, and then generates Violation code for each raw APSR by modifying the Right calling code using LLM. Subsequently, GPTAid performs dynamic execution on each piece of Violation code and further filters out the incorrect APSRs based on runtime errors. To further generate concrete APSRs, GPTAid employs a code differential analysis to refine the filtered ones. Particularly, as the programming language is more precise than natural language, GPTAid identifies the key operations within Violation code by differential analysis, and then generates the corresponding concrete APSR based on the aforementioned operations. These concrete APSRs could be precisely interpreted into applicable detection code, which proven to be effective in API misuse detection. Implementing on the dataset containing 200 randomly selected APIs from eight popular libraries, GPTAid achieves a precision of 92.3%. Moreover, it generates 6 times more APSRs than state-of-the-art detectors on a comparison dataset of previously reported bugs and APSRs. We further evaluated GPTAid on 47 applications, 210 unknown security bugs were found potentially resulting in severe security issues (e.g., system crashes), 150 of which have been confirmed by developers after our reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11690v2">LLM-Powered Text Simulation Attack Against ID-Free Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2024-09-19
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      The ID-free recommendation paradigm has been proposed to address the limitation that traditional recommender systems struggle to model cold-start users or items with new IDs. Despite its effectiveness, this study uncovers that ID-free recommender systems are vulnerable to the proposed Text Simulation attack (TextSimu) which aims to promote specific target items. As a novel type of text poisoning attack, TextSimu exploits large language models (LLM) to alter the textual information of target items by simulating the characteristics of popular items. It operates effectively in both black-box and white-box settings, utilizing two key components: a unified popularity extraction module, which captures the essential characteristics of popular items, and an N-persona consistency simulation strategy, which creates multiple personas to collaboratively synthesize refined promotional textual descriptions for target items by simulating the popular items. To withstand TextSimu-like attacks, we further explore the detection approach for identifying LLM-generated promotional text. Extensive experiments conducted on three datasets demonstrate that TextSimu poses a more significant threat than existing poisoning attacks, while our defense method can detect malicious text of target items generated by TextSimu. By identifying the vulnerability, we aim to advance the development of more robust ID-free recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12411v1">Textualized Agent-Style Reasoning for Complex Tasks by Multiple Round LLM Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      Chain-of-thought prompting significantly boosts the reasoning ability of large language models but still faces three issues: hallucination problem, restricted interpretability, and uncontrollable generation. To address these challenges, we present AgentCOT, a llm-based autonomous agent framework, which can solve complex problems in an agent-style manner by multiple round LLM generation. At each step, AgentCOT selects an action and executes it to yield an intermediate result with supporting evidence. In addition, we integrate the step's index into the reasoning process to form a graph structure for complex inference logic. We introduce two new strategies to enhance the performance of AgentCOT.We conduct extensive experiments to verify the effectiveness of our method on six common benchmarks. Results exhibit that our method brings in substantial improvements over current competitive approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12405v1">On the Effectiveness of LLMs for Manual Test Verifications</a></div>
    <div class="paper-meta">
      📅 2024-09-19
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Background: Manual testing is vital for detecting issues missed by automated tests, but specifying accurate verifications is challenging. Aims: This study aims to explore the use of Large Language Models (LLMs) to produce verifications for manual tests. Method: We conducted two independent and complementary exploratory studies. The first study involved using 2 closed-source and 6 open-source LLMs to generate verifications for manual test steps and evaluate their similarity to original verifications. The second study involved recruiting software testing professionals to assess their perception and agreement with the generated verifications compared to the original ones. Results: The open-source models Mistral-7B and Phi-3-mini-4k demonstrated effectiveness and consistency comparable to closed-source models like Gemini-1.5-flash and GPT-3.5-turbo in generating manual test verifications. However, the agreement level among professional testers was slightly above 40%, indicating both promise and room for improvement. While some LLM-generated verifications were considered better than the originals, there were also concerns about AI hallucinations, where verifications significantly deviated from expectations. Conclusion: We contributed by generating a dataset of 37,040 test verifications using 8 different LLMs. Although the models show potential, the relatively modest 40% agreement level highlights the need for further refinement. Enhancing the accuracy, relevance, and clarity of the generated verifications is crucial to ensure greater reliability in real-world testing scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00225v2">Large-scale, Independent and Comprehensive study of the power of LLMs for test case generation</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      Unit testing, crucial for ensuring the reliability of code modules, such as classes and methods, is often overlooked by developers due to time constraints. Automated test generation techniques have emerged to address this, but they frequently lack readability and require significant developer intervention. Large Language Models (LLMs), such as GPT and Mistral, have shown promise in software engineering tasks, including test generation, but their overall effectiveness remains unclear. This study presents an extensive investigation of LLMs, evaluating the effectiveness of four models and five prompt engineering techniques for unit test generation. We analyze 216 300 tests generated by the selected advanced instruct-tuned LLMs for 690 Java classes collected from diverse datasets. Our evaluation considers correctness, understandability, coverage, and test smell detection in the generated tests, comparing them to a widely used automated testing tool, EvoSuite. While LLMs demonstrate potential, improvements in test quality particularly in reducing common test smells are necessary. This study highlights the strengths and limitations of LLM-generated tests compared to traditional methods, paving the way for further research on LLMs in test automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10011v2">HALO: Hallucination Analysis and Learning Optimization to Empower LLMs with Retrieval-Augmented Context for Guided Clinical Decision Making</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced natural language processing tasks, yet they are susceptible to generating inaccurate or unreliable responses, a phenomenon known as hallucination. In critical domains such as health and medicine, these hallucinations can pose serious risks. This paper introduces HALO, a novel framework designed to enhance the accuracy and reliability of medical question-answering (QA) systems by focusing on the detection and mitigation of hallucinations. Our approach generates multiple variations of a given query using LLMs and retrieves relevant information from external open knowledge bases to enrich the context. We utilize maximum marginal relevance scoring to prioritize the retrieved context, which is then provided to LLMs for answer generation, thereby reducing the risk of hallucinations. The integration of LangChain further streamlines this process, resulting in a notable and robust increase in the accuracy of both open-source and commercial LLMs, such as Llama-3.1 (from 44% to 65%) and ChatGPT (from 56% to 70%). This framework underscores the critical importance of addressing hallucinations in medical QA systems, ultimately improving clinical decision-making and patient care. The open-source HALO is available at: https://github.com/ResponsibleAILab/HALO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12274v1">Hierarchical LLMs In-the-loop Optimization for Real-time Multi-Robot Target Tracking under Unknown Hazards</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      In this paper, we propose a hierarchical Large Language Models (LLMs) in-the-loop optimization framework for real-time multi-robot task allocation and target tracking in an unknown hazardous environment subject to sensing and communication attacks. We formulate multi-robot coordination for tracking tasks as a bi-level optimization problem, with LLMs to reason about potential hazards in the environment and the status of the robot team and modify both the inner and outer levels of the optimization. The inner LLM adjusts parameters to prioritize various objectives, including performance, safety, and energy efficiency, while the outer LLM handles online variable completion for team reconfiguration. This hierarchical approach enables real-time adjustments to the robots' behavior. Additionally, a human supervisor can offer broad guidance and assessments to address unexpected dangers, model mismatches, and performance issues arising from local minima. We validate our proposed framework in both simulation and real-world experiments with comprehensive evaluations, which provide the potential for safe LLM integration for multi-robot problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12150v1">Decoding Style: Efficient Fine-Tuning of LLMs for Image-Guided Outfit Recommendation with Preference</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 CIKM 2024
    </div>
    <details class="paper-abstract">
      Personalized outfit recommendation remains a complex challenge, demanding both fashion compatibility understanding and trend awareness. This paper presents a novel framework that harnesses the expressive power of large language models (LLMs) for this task, mitigating their "black box" and static nature through fine-tuning and direct feedback integration. We bridge the item visual-textual gap in items descriptions by employing image captioning with a Multimodal Large Language Model (MLLM). This enables the LLM to extract style and color characteristics from human-curated fashion images, forming the basis for personalized recommendations. The LLM is efficiently fine-tuned on the open-source Polyvore dataset of curated fashion images, optimizing its ability to recommend stylish outfits. A direct preference mechanism using negative examples is employed to enhance the LLM's decision-making process. This creates a self-enhancing AI feedback loop that continuously refines recommendations in line with seasonal fashion trends. Our framework is evaluated on the Polyvore dataset, demonstrating its effectiveness in two key tasks: fill-in-the-blank, and complementary item retrieval. These evaluations underline the framework's ability to generate stylish, trend-aligned outfit suggestions, continuously improving through direct feedback. The evaluation results demonstrated that our proposed framework significantly outperforms the base LLM, creating more cohesive outfits. The improved performance in these tasks underscores the proposed framework's potential to enhance the shopping experience with accurate suggestions, proving its effectiveness over the vanilla LLM based outfit generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03556v1">BodyShapeGPT: SMPL Body Shape Manipulation with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Accepted to ECCV 2024 Workshop on Foundation Models for 3D Humans. Code repository: https://github.com/baldoarbol/BodyShapeGPT
    </div>
    <details class="paper-abstract">
      Generative AI models provide a wide range of tools capable of performing complex tasks in a fraction of the time it would take a human. Among these, Large Language Models (LLMs) stand out for their ability to generate diverse texts, from literary narratives to specialized responses in different fields of knowledge. This paper explores the use of fine-tuned LLMs to identify physical descriptions of people, and subsequently create accurate representations of avatars using the SMPL-X model by inferring shape parameters. We demonstrate that LLMs can be trained to understand and manipulate the shape space of SMPL, allowing the control of 3D human shapes through natural language. This approach promises to improve human-machine interaction and opens new avenues for customization and simulation in virtual environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12117v1">Low Frame-rate Speech Codec: a Codec Designed for Fast High-quality Speech LLM Training and Inference</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Submitted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced audio processing through audio codecs that convert audio into discrete tokens, enabling the application of language modeling techniques to audio data. However, audio codecs often operate at high frame rates, resulting in slow training and inference, especially for autoregressive models. To address this challenge, we present the Low Frame-rate Speech Codec (LFSC): a neural audio codec that leverages finite scalar quantization and adversarial training with large speech language models to achieve high-quality audio compression with a 1.89 kbps bitrate and 21.5 frames per second. We demonstrate that our novel codec can make the inference of LLM-based text-to-speech models around three times faster while improving intelligibility and producing quality comparable to previous models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19736v1">Combining LLM Code Generation with Formal Specifications and Reactive Program Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      In the past few years, Large Language Models (LLMs) have exploded in usefulness and popularity for code generation tasks. However, LLMs still struggle with accuracy and are unsuitable for high-risk applications without additional oversight and verification. In particular, they perform poorly at generating code for highly complex systems, especially with unusual or out-of-sample logic. For such systems, verifying the code generated by the LLM may take longer than writing it by hand. We introduce a solution that divides the code generation into two parts; one to be handled by an LLM and one to be handled by formal methods-based program synthesis. We develop a benchmark to test our solution and show that our method allows the pipeline to solve problems previously intractable for LLM code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11971v1">Sampling Latent Material-Property Information From LLM-Derived Embedding Representations</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Vector embeddings derived from large language models (LLMs) show promise in capturing latent information from the literature. Interestingly, these can be integrated into material embeddings, potentially useful for data-driven predictions of materials properties. We investigate the extent to which LLM-derived vectors capture the desired information and their potential to provide insights into material properties without additional training. Our findings indicate that, although LLMs can be used to generate representations reflecting certain property information, extracting the embeddings requires identifying the optimal contextual clues and appropriate comparators. Despite this restriction, it appears that LLMs still have the potential to be useful in generating meaningful materials-science representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11917v1">LLMs in Education: Novel Perspectives, Challenges, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 COLING 2025 Tutorial
    </div>
    <details class="paper-abstract">
      The role of large language models (LLMs) in education is an increasing area of interest today, considering the new opportunities they offer for teaching, learning, and assessment. This cutting-edge tutorial provides an overview of the educational applications of NLP and the impact that the recent advances in LLMs have had on this field. We will discuss the key challenges and opportunities presented by LLMs, grounding them in the context of four major educational applications: reading, writing, and speaking skills, and intelligent tutoring systems (ITS). This COLING 2025 tutorial is designed for researchers and practitioners interested in the educational applications of NLP and the role LLMs have to play in this area. It is the first of its kind to address this timely topic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11901v1">LLMs + Persona-Plug = Personalized LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      Personalization plays a critical role in numerous language tasks and applications, since users with the same requirements may prefer diverse outputs based on their individual interests. This has led to the development of various personalized approaches aimed at adapting large language models (LLMs) to generate customized outputs aligned with user preferences. Some of them involve fine-tuning a unique personalized LLM for each user, which is too expensive for widespread application. Alternative approaches introduce personalization information in a plug-and-play manner by retrieving the user's relevant historical texts as demonstrations. However, this retrieval-based strategy may break the continuity of the user history and fail to capture the user's overall styles and patterns, hence leading to sub-optimal performance. To address these challenges, we propose a novel personalized LLM model, \ours{}. It constructs a user-specific embedding for each individual by modeling all her historical contexts through a lightweight plug-in user embedder module. By attaching this embedding to the task input, LLMs can better understand and capture user habits and preferences, thereby producing more personalized outputs without tuning their own parameters. Extensive experiments on various tasks in the language model personalization (LaMP) benchmark demonstrate that the proposed model significantly outperforms existing personalized LLM approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11860v1">Retrieve, Annotate, Evaluate, Repeat: Leveraging Multimodal LLMs for Large-Scale Product Retrieval Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 13 pages, 5 figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Evaluating production-level retrieval systems at scale is a crucial yet challenging task due to the limited availability of a large pool of well-trained human annotators. Large Language Models (LLMs) have the potential to address this scaling issue and offer a viable alternative to humans for the bulk of annotation tasks. In this paper, we propose a framework for assessing the product search engines in a large-scale e-commerce setting, leveraging Multimodal LLMs for (i) generating tailored annotation guidelines for individual queries, and (ii) conducting the subsequent annotation task. Our method, validated through deployment on a large e-commerce platform, demonstrates comparable quality to human annotations, significantly reduces time and cost, facilitates rapid problem discovery, and provides an effective solution for production-level quality control at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13764v1">Local Explanations and Self-Explanations for Assessing Faithfulness in black-box LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      This paper introduces a novel task to assess the faithfulness of large language models (LLMs) using local perturbations and self-explanations. Many LLMs often require additional context to answer certain questions correctly. For this purpose, we propose a new efficient alternative explainability technique, inspired by the commonly used leave-one-out approach. Using this approach, we identify the sufficient and necessary parts for the LLM to generate correct answers, serving as explanations. We propose a metric for assessing faithfulness that compares these crucial parts with the self-explanations of the model. Using the Natural Questions dataset, we validate our approach, demonstrating its effectiveness in explaining model decisions and assessing faithfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11844v1">MEOW: MEMOry Supervised LLM Unlearning Via Inverted Facts</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can memorize sensitive information, raising concerns about potential misuse. LLM Unlearning, a post-hoc approach to remove this information from trained LLMs, offers a promising solution to mitigate these risks. However, previous practices face three key challenges: 1. Utility: successful unlearning often causes catastrophic collapse on unrelated tasks. 2. Efficiency: many methods either involve adding similarly sized models, which slows down unlearning or inference, or require retain data that are difficult to obtain. 3. Robustness: even effective methods may still leak data via extraction techniques. To address these challenges, we propose MEOW, a simple yet effective gradient descent-based unlearning method. Specifically, we use an offline LLM to generate a set of inverted facts. Then, we design a new metric, MEMO, to quantify memorization in LLMs. Finally, based on the signals provided by MEMO, we select the most appropriate set of inverted facts and finetune the model based on them. We evaluate MEOW on the commonly used unlearn benchmark, ToFU, with Llama2-7B-Chat and Phi-1.5B, and test it on both NLU and NLG tasks. Results demonstrate significant improvement of MEOW in forget quality without substantial loss in model utility. Meanwhile, MEOW does not exhibit significant degradation in NLU or NLG capabilities, and there is even a slight improvement in NLU performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10697v2">LLMs as information warriors? Auditing how LLM-powered chatbots tackle disinformation about Russia's war in Ukraine</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has a significant impact on information warfare. By facilitating the production of content related to disinformation and propaganda campaigns, LLMs can amplify different types of information operations and mislead online users. In our study, we empirically investigate how LLM-powered chatbots, developed by Google, Microsoft, and Perplexity, handle disinformation about Russia's war in Ukraine and whether the chatbots' ability to provide accurate information on the topic varies across languages and over time. Our findings indicate that while for some chatbots (Perplexity), there is a significant improvement in performance over time in several languages, for others (Gemini), the performance improves only in English but deteriorates in low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15361v1">Multitask Mayhem: Unveiling and Mitigating Safety Gaps in LLMs Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 19 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in Large Language Models (LLMs) have led to their adoption across a wide range of tasks, ranging from code generation to machine translation and sentiment analysis, etc. Red teaming/Safety alignment efforts show that fine-tuning models on benign (non-harmful) data could compromise safety. However, it remains unclear to what extent this phenomenon is influenced by different variables, including fine-tuning task, model calibrations, etc. This paper explores the task-wise safety degradation due to fine-tuning on downstream tasks such as summarization, code generation, translation, and classification across various calibration. Our results reveal that: 1) Fine-tuning LLMs for code generation and translation leads to the highest degradation in safety guardrails. 2) LLMs generally have weaker guardrails for translation and classification, with 73-92% of harmful prompts answered, across baseline and other calibrations, falling into one of two concern categories. 3) Current solutions, including guards and safety tuning datasets, lack cross-task robustness. To address these issues, we developed a new multitask safety dataset effectively reducing attack success rates across a range of tasks without compromising the model's overall helpfulness. Our work underscores the need for generalized alignment measures to ensure safer and more robust models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06787v2">Unlock the Power of Frozen LLMs in Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      Traditional knowledge graph completion (KGC) methods rely solely on structural information, struggling with the inherent sparsity of knowledge graphs (KGs). Large Language Models (LLMs) learn extensive knowledge from large corpora with powerful context modeling, making them promising for mitigating the limitations of previous methods. Directly fine-tuning LLMs offers great capability but comes at the cost of huge time and memory consumption, while utilizing frozen LLMs yields suboptimal results.In this work, we aim to leverage LLMs for KGC effectively and efficiently. We capture the context-aware hidden states of knowledge triples by employing prompts to stimulate the intermediate layers of LLMs. We then train a data-efficient classifier on these hidden states to harness the inherent capabilities of frozen LLMs in KGC. Additionally, to reduce ambiguity and enrich knowledge representation, we generate detailed entity descriptions through subgraph sampling on KGs. Extensive experiments on standard benchmarks demonstrate the efficiency and effectiveness of our approach. We outperform traditional KGC methods across most datasets and, notably, achieve classification performance comparable to fine-tuned LLMs while enhancing GPU memory efficiency by $188\times$ and accelerating training and inference by $13.48\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07970v3">Guiding In-Context Learning of LLMs through Quality Estimation for Machine Translation</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Camera-ready version of the paper for the Association for Machine Translation in the Americas (AMTA), including the link to the paper's repository
    </div>
    <details class="paper-abstract">
      The quality of output from large language models (LLMs), particularly in machine translation (MT), is closely tied to the quality of in-context examples (ICEs) provided along with the query, i.e., the text to translate. The effectiveness of these ICEs is influenced by various factors, such as the domain of the source text, the order in which the ICEs are presented, the number of these examples, and the prompt templates used. Naturally, selecting the most impactful ICEs depends on understanding how these affect the resulting translation quality, which ultimately relies on translation references or human judgment. This paper presents a novel methodology for in-context learning (ICL) that relies on a search algorithm guided by domain-specific quality estimation (QE). Leveraging the XGLM model, our methodology estimates the resulting translation quality without the need for translation references, selecting effective ICEs for MT to maximize translation quality. Our results demonstrate significant improvements over existing ICL methods and higher translation performance compared to fine-tuning a pre-trained language model (PLM), specifically mBART-50.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11726v1">Revealing the Challenge of Detecting Character Knowledge Errors in LLM Role-Playing</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 22 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) role-playing has gained widespread attention, where the authentic character knowledge is crucial for constructing realistic LLM role-playing agents. However, existing works usually overlook the exploration of LLMs' ability to detect characters' known knowledge errors (KKE) and unknown knowledge errors (UKE) while playing roles, which would lead to low-quality automatic construction of character trainable corpus. In this paper, we propose a probing dataset to evaluate LLMs' ability to detect errors in KKE and UKE. The results indicate that even the latest LLMs struggle to effectively detect these two types of errors, especially when it comes to familiar knowledge. We experimented with various reasoning strategies and propose an agent-based reasoning method, Self-Recollection and Self-Doubt (S2RD), to further explore the potential for improving error detection capabilities. Experiments show that our method effectively improves the LLMs' ability to detect error character knowledge, but it remains an issue that requires ongoing attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11703v1">Harnessing LLMs for API Interactions: A Framework for Classification and Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) advance in natural language processing, there is growing interest in leveraging their capabilities to simplify software interactions. In this paper, we propose a novel system that integrates LLMs for both classifying natural language inputs into corresponding API calls and automating the creation of sample datasets tailored to specific API functions. By classifying natural language commands, our system allows users to invoke complex software functionalities through simple inputs, improving interaction efficiency and lowering the barrier to software utilization. Our dataset generation approach also enables the efficient and systematic evaluation of different LLMs in classifying API calls, offering a practical tool for developers or business owners to assess the suitability of LLMs for customized API management. We conduct experiments on several prominent LLMs using generated sample datasets for various API functions. The results show that GPT-4 achieves a high classification accuracy of 0.996, while LLaMA-3-8B performs much worse at 0.759. These findings highlight the potential of LLMs to transform API management and validate the effectiveness of our system in guiding model testing and selection across diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09044v2">MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      Efficient finetuning of large language models (LLMs) aims to adapt the LLMs with reduced computational and memory cost. Previous LoRA-based approaches initialize the low-rank matrices with Gaussian distribution and zero values while keeping the original weight matrices frozen. However, the trainable model parameters optimized in an unguided subspace might interfere with the well-learned subspace of the pretrained weight matrices. In this paper, we propose MiLoRA, a simple yet effective LLM finetuning approach that only updates the minor singular components of the weight matrix while keeping the principal singular components frozen. It is observed that the minor matrix corresponds to the noisy or long-tail information, while the principal matrix contains important knowledge. The MiLoRA initializes the low-rank matrices within a subspace that is orthogonal to the principal matrix, thus the pretrained knowledge is expected to be well preserved. During finetuning, MiLoRA makes the most use of the less-optimized subspace for learning the labeled dataset. Extensive experiments on commonsense reasoning, math reasoning, instruction following and visual instruction following benchmarks present the superior performance of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11638v1">BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      This study presents BanStereoSet, a dataset designed to evaluate stereotypical social biases in multilingual LLMs for the Bangla language. In an effort to extend the focus of bias research beyond English-centric datasets, we have localized the content from the StereoSet, IndiBias, and Kamruzzaman et. al.'s datasets, producing a resource tailored to capture biases prevalent within the Bangla-speaking community. Our BanStereoSet dataset consists of 1,194 sentences spanning 9 categories of bias: race, profession, gender, ageism, beauty, beauty in profession, region, caste, and religion. This dataset not only serves as a crucial tool for measuring bias in multilingual LLMs but also facilitates the exploration of stereotypical bias across different social categories, potentially guiding the development of more equitable language technologies in Bangladeshi contexts. Our analysis of several language models using this dataset indicates significant biases, reinforcing the necessity for culturally and linguistically adapted datasets to develop more equitable language technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11636v1">"A Woman is More Culturally Knowledgeable than A Man?": The Effect of Personas on Cultural Norm Interpretation in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Preprint, Under Review
    </div>
    <details class="paper-abstract">
      As the deployment of large language models (LLMs) expands, there is an increasing demand for personalized LLMs. One method to personalize and guide the outputs of these models is by assigning a persona -- a role that describes the expected behavior of the LLM (e.g., a man, a woman, an engineer). This study investigates whether an LLM's understanding of social norms varies across assigned personas. Ideally, the perception of a social norm should remain consistent regardless of the persona, since acceptability of a social norm should be determined by the region the norm originates from, rather than by individual characteristics such as gender, body size, or race. A norm is universal within its cultural context. In our research, we tested 36 distinct personas from 12 sociodemographic categories (e.g., age, gender, beauty) across four different LLMs. We find that LLMs' cultural norm interpretation varies based on the persona used and the norm interpretation also varies within a sociodemographic category (e.g., a fat person and a thin person as in physical appearance group) where an LLM with the more socially desirable persona (e.g., a thin person) interprets social norms more accurately than with the less socially desirable persona (e.g., a fat person). We also discuss how different types of social biases may contribute to the results that we observe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08221v2">FAIL: Analyzing Software Failures from the News Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Accapted at the 9th IEEE/ACM International Conference on Automated Software Engineering (ASE 2024)
    </div>
    <details class="paper-abstract">
      Software failures inform engineering work, standards, regulations. For example, the Log4J vulnerability brought government and industry attention to evaluating and securing software supply chains. Accessing private engineering records is difficult, so failure analyses tend to use information reported by the news media. However, prior works in this direction have relied on manual analysis. That has limited the scale of their analyses. The community lacks automated support to enable such analyses to consider a wide range of news sources and incidents. In this paper, we propose the Failure Analysis Investigation with LLMs (FAIL) system to fill this gap. FAIL collects, analyzes, and summarizes software failures as reported in the news. FAIL groups articles that describe the same incidents. It then analyzes incidents using existing taxonomies for postmortems, faults, and system characteristics. To tune and evaluate FAIL, we followed the methods of prior works by manually analyzing 31 software failures. FAIL achieved an F1 score of 90% for collecting news about software failures, a V-measure of 0.98 for merging articles reporting on the same incident, and extracted 90% of the facts about failures. We then applied FAIL to a total of 137,427 news articles from 11 providers published between 2010 and 2022. FAIL identified and analyzed 2457 distinct failures reported across 4,184 articles. Our findings include: (1) current generation of large language models are capable of identifying news articles that describe failures, and analyzing them according to structured taxonomies; (2) high recurrences of similar failures within organizations and across organizations; and (3) severity of the consequences of software failures have increased over the past decade. The full FAIL database is available so that researchers, engineers, and policymakers can learn from a diversity of software failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11580v1">PLATO: Planning with LLMs and Affordances for Tool Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-09-17
      | 💬 7 pages, 4 figures, submitted to ICRA 2025
    </div>
    <details class="paper-abstract">
      As robotic systems become increasingly integrated into complex real-world environments, there is a growing need for approaches that enable robots to understand and act upon natural language instructions without relying on extensive pre-programmed knowledge of their surroundings. This paper presents PLATO, an innovative system that addresses this challenge by leveraging specialized large language model agents to process natural language inputs, understand the environment, predict tool affordances, and generate executable actions for robotic systems. Unlike traditional systems that depend on hard-coded environmental information, PLATO employs a modular architecture of specialized agents to operate without any initial knowledge of the environment. These agents identify objects and their locations within the scene, generate a comprehensive high-level plan, translate this plan into a series of low-level actions, and verify the completion of each step. The system is particularly tested on challenging tool-use tasks, which involve handling diverse objects and require long-horizon planning. PLATO's design allows it to adapt to dynamic and unstructured settings, significantly enhancing its flexibility and robustness. By evaluating the system across various complex scenarios, we demonstrate its capability to tackle a diverse range of tasks and offer a novel solution to integrate LLMs with robotic platforms, advancing the state-of-the-art in autonomous robotic task execution. For videos and prompt details, please see our project website: https://sites.google.com/andrew.cmu.edu/plato
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11276v1">Hackphyr: A Local Fine-Tuned LLM Agent for Network Security Environments</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable potential across various domains, including cybersecurity. Using commercial cloud-based LLMs may be undesirable due to privacy concerns, costs, and network connectivity constraints. In this paper, we present Hackphyr, a locally fine-tuned LLM to be used as a red-team agent within network security environments. Our fine-tuned 7 billion parameter model can run on a single GPU card and achieves performance comparable with much larger and more powerful commercial models such as GPT-4. Hackphyr clearly outperforms other models, including GPT-3.5-turbo, and baselines, such as Q-learning agents in complex, previously unseen scenarios. To achieve this performance, we generated a new task-specific cybersecurity dataset to enhance the base model's capabilities. Finally, we conducted a comprehensive analysis of the agents' behaviors that provides insights into the planning abilities and potential shortcomings of such agents, contributing to the broader understanding of LLM-based agents in cybersecurity contexts
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11214v1">Ideal-LLM: Integrating Dual Encoders and Language-Adapted LLM for Multilingual Speech-to-Text</a></div>
    <div class="paper-meta">
      📅 2024-09-17
      | 💬 5 pages, 3 figures, submitted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      Integrating audio encoders with LLMs through connectors has enabled these models to process and comprehend audio modalities, significantly enhancing speech-to-text tasks, including automatic speech recognition (ASR) and automatic speech translation (AST). However, these methods often overlook the critical aspect of language adaptation in multilingual settings, relying instead on multilingual data without adequately addressing language differences. To address this gap, we propose the Ideal-LLM model, which employs dual multilingual encoders to enrich language feature information and utilizes a language-adapted connector to target the adaptation of each language specifically. By leveraging the complementary strengths of Whisper and MMS encoders, our approach ensures richer multilingual representations. Additionally, the language-adapted connector enhances modal transformation via a language weight selector tailored for each language. Experimental results demonstrate that Ideal-LLM significantly improves ASR performance, achieving a 32.6% relative reduction in average word error rates compared to the standard speech encoder integrated with LLMs and yields an average BLEU score of 36.78 for AST task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11056v1">Large Language Models are Good Multi-lingual Learners : When LLMs Meet Cross-lingual Prompts</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      With the advent of Large Language Models (LLMs), generating rule-based data for real-world applications has become more accessible. Due to the inherent ambiguity of natural language and the complexity of rule sets, especially in long contexts, LLMs often struggle to follow all specified rules, frequently omitting at least one. To enhance the reasoning and understanding of LLMs on long and complex contexts, we propose a novel prompting strategy Multi-Lingual Prompt, namely MLPrompt, which automatically translates the error-prone rule that an LLM struggles to follow into another language, thus drawing greater attention to it. Experimental results on public datasets across various tasks have shown MLPrompt can outperform state-of-the-art prompting methods such as Chain of Thought, Tree of Thought, and Self-Consistency. Additionally, we introduce a framework integrating MLPrompt with an auto-checking mechanism for structured data generation, with a specific case study in text-to-MIP instances. Further, we extend the proposed framework for text-to-SQL to demonstrate its generation ability towards structured data synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10969v1">Enhancing Multilingual Speech Generation and Recognition Abilities in LLMs with Constructed Code-switched Data</a></div>
    <div class="paper-meta">
      📅 2024-09-17
      | 💬 Submitted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have been explored in the speech domain for both generation and recognition tasks, their applications are predominantly confined to the monolingual scenario, with limited exploration in multilingual and code-switched (CS) contexts. Additionally, speech generation and recognition tasks are often handled separately, such as VALL-E and Qwen-Audio. In this paper, we propose a MutltiLingual MultiTask (MLMT) model, integrating multilingual speech generation and recognition tasks within the single LLM. Furthermore, we develop an effective data construction approach that splits and concatenates words from different languages to equip LLMs with CS synthesis ability without relying on CS data. The experimental results demonstrate that our model outperforms other baselines with a comparable data scale. Furthermore, our data construction approach not only equips LLMs with CS speech synthesis capability with comparable speaker consistency and similarity to any given speaker, but also improves the performance of LLMs in multilingual speech generation and recognition tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10913v1">ASHABot: An LLM-Powered Chatbot to Support the Informational Needs of Community Health Workers</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      Community health workers (CHWs) provide last-mile healthcare services but face challenges due to limited medical knowledge and training. This paper describes the design, deployment, and evaluation of ASHABot, an LLM-powered, experts-in-the-loop, WhatsApp-based chatbot to address the information needs of CHWs in India. Through interviews with CHWs and their supervisors and log analysis, we examine factors affecting their engagement with ASHABot, and ASHABot's role in addressing CHWs' informational needs. We found that ASHABot provided a private channel for CHWs to ask rudimentary and sensitive questions they hesitated to ask supervisors. CHWs trusted the information they received on ASHABot and treated it as an authoritative resource. CHWs' supervisors expanded their knowledge by contributing answers to questions ASHABot failed to answer, but were concerned about demands on their workload and increased accountability. We emphasize positioning LLMs as supplemental fallible resources within the community healthcare ecosystem, instead of as replacements for supervisor support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02768v1">BoViLA: Bootstrapping Video-Language Alignment via LLM-Based Self-Questioning and Answering</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      The development of multi-modal models has been rapidly advancing, with some demonstrating remarkable capabilities. However, annotating video-text pairs remains expensive and insufficient. Take video question answering (VideoQA) tasks as an example, human annotated questions and answers often cover only part of the video, and similar semantics can also be expressed through different text forms, leading to underutilization of video. To address this, we propose BoViLA, a self-training framework that augments question samples during training through LLM-based self-questioning and answering, which help model exploit video information and the internal knowledge of LLMs more thoroughly to improve modality alignment. To filter bad self-generated questions, we introduce Evidential Deep Learning (EDL) to estimate uncertainty and assess the quality of self-generated questions by evaluating the modality alignment within the context. To the best of our knowledge, this work is the first to explore LLM-based self-training frameworks for modality alignment. We evaluate BoViLA on five strong VideoQA benchmarks, where it outperforms several state-of-the-art methods and demonstrate its effectiveness and generality. Additionally, we provide extensive analyses of the self-training framework and the EDL-based uncertainty filtering mechanism. The code will be made available at https://github.com/dunknsabsw/BoViLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10354v2">Learnings from a Large-Scale Deployment of an LLM-Powered Expert-in-the-Loop Healthcare Chatbot</a></div>
    <div class="paper-meta">
      📅 2024-09-17
      | 💬 The first two authors contributed equally to this research
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in healthcare, but limitations like hallucinations, incomplete information, and bias hinder their reliability. To address these, researchers released the Build Your Own expert Bot (BYOeB) platform, enabling developers to create LLM-powered chatbots with integrated expert verification. CataractBot, its first implementation, provides expert-verified responses to cataract surgery questions. A pilot evaluation showed its potential; however the study had a small sample size and was primarily qualitative. In this work, we conducted a large-scale 24-week deployment of CataractBot involving 318 patients and attendants who sent 1,992 messages, with 91.71% of responses verified by seven experts. Analysis of interaction logs revealed that medical questions significantly outnumbered logistical ones, hallucinations were negligible, and experts rated 84.52% of medical answers as accurate. As the knowledge base expanded with expert corrections, system performance improved by 19.02%, reducing expert workload. These insights guide the design of future LLM-powered chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09383v2">LLM-Powered Ensemble Learning for Paper Source Tracing: A GPU-Free Approach</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      We participated in the KDD CUP 2024 paper source tracing competition and achieved the 3rd place. This competition tasked participants with identifying the reference sources (i.e., ref-sources, as referred to by the organizers of the competition) of given academic papers. Unlike most teams that addressed this challenge by fine-tuning pre-trained neural language models such as BERT or ChatGLM, our primary approach utilized closed-source large language models (LLMs). With recent advancements in LLM technology, closed-source LLMs have demonstrated the capability to tackle complex reasoning tasks in zero-shot or few-shot scenarios. Consequently, in the absence of GPUs, we employed closed-source LLMs to directly generate predicted reference sources from the provided papers. We further refined these predictions through ensemble learning. Notably, our method was the only one among the award-winning approaches that did not require the use of GPUs for model training. Code available at https://github.com/Cklwanfifa/KDDCUP2024-PST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08330v2">Real or Robotic? Assessing Whether LLMs Accurately Simulate Qualities of Human Responses in Dialogue</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      Studying and building datasets for dialogue tasks is both expensive and time-consuming due to the need to recruit, train, and collect data from study participants. In response, much recent work has sought to use large language models (LLMs) to simulate both human-human and human-LLM interactions, as they have been shown to generate convincingly human-like text in many settings. However, to what extent do LLM-based simulations \textit{actually} reflect human dialogues? In this work, we answer this question by generating a large-scale dataset of 100,000 paired LLM-LLM and human-LLM dialogues from the WildChat dataset and quantifying how well the LLM simulations align with their human counterparts. Overall, we find relatively low alignment between simulations and human interactions, demonstrating a systematic divergence along the multiple textual properties, including style and content. Further, in comparisons of English, Chinese, and Russian dialogues, we find that models perform similarly. Our results suggest that LLMs generally perform better when the human themself writes in a way that is more similar to the LLM's own style.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10444v1">LLM as BT-Planner: Leveraging LLMs for Behavior Tree Generation in Robot Task Planning</a></div>
    <div class="paper-meta">
      📅 2024-09-16
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Robotic assembly tasks are open challenges due to the long task horizon and complex part relations. Behavior trees (BTs) are increasingly used in robot task planning for their modularity and flexibility, but manually designing them can be effort-intensive. Large language models (LLMs) have recently been applied in robotic task planning for generating action sequences, but their ability to generate BTs has not been fully investigated. To this end, We propose LLM as BT-planner, a novel framework to leverage LLMs for BT generation in robotic assembly task planning and execution. Four in-context learning methods are introduced to utilize the natural language processing and inference capabilities of LLMs to produce task plans in BT format, reducing manual effort and ensuring robustness and comprehensibility. We also evaluate the performance of fine-tuned, fewer-parameter LLMs on the same tasks. Experiments in simulated and real-world settings show that our framework enhances LLMs' performance in BT generation, improving success rates in BT generation through in-context learning and supervised fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01824v1">AI Conversational Interviewing: Transforming Surveys with LLMs as Adaptive Interviewers</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      Traditional methods for eliciting people's opinions face a trade-off between depth and scale: structured surveys enable large-scale data collection but limit respondents' ability to express unanticipated thoughts in their own words, while conversational interviews provide deeper insights but are resource-intensive. This study explores the potential of replacing human interviewers with large language models (LLMs) to conduct scalable conversational interviews. Our goal is to assess the performance of AI Conversational Interviewing and to identify opportunities for improvement in a controlled environment. We conducted a small-scale, in-depth study with university students who were randomly assigned to be interviewed by either AI or human interviewers, both employing identical questionnaires on political topics. Various quantitative and qualitative measures assessed interviewer adherence to guidelines, response quality, participant engagement, and overall interview efficacy. The findings indicate the viability of AI Conversational Interviewing in producing quality data comparable to traditional methods, with the added benefit of scalability. Based on our experiences, we present specific recommendations for effective implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11477v2">How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in many languages beyond English. Yet, LLMs require more inference steps when generating non-English text due to their reliance on English-centric tokenizers and vocabulary, resulting in higher usage costs to non-English speakers. Vocabulary expansion with target language tokens is a widely used cross-lingual vocabulary adaptation approach to remedy this issue. Despite its effectiveness in inference speedup, previous work on vocabulary expansion has focused on high-resource settings assuming access to a substantial amount of target language data to effectively initialize the embeddings of the new tokens and adapt the LLM to the target language. However, vocabulary expansion in low-resource settings has yet to be explored. In this paper, we investigate vocabulary expansion in low-resource settings by considering embedding initialization methods and continual pre-training strategies. Through extensive experiments across typologically diverse languages, tasks and models, we establish a set of strategies to perform vocabulary expansion for faster inference, maintaining competitive downstream performance to baselines with only 30K sentences ($\sim$0.01GB text data) from the target language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15051v3">Prior Knowledge Integration via LLM Encoding and Pseudo Event Regulation for Video Moment Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-09-16
      | 💬 Accepted to ACM Multimedia 2024
    </div>
    <details class="paper-abstract">
      In this paper, we investigate the feasibility of leveraging large language models (LLMs) for integrating general knowledge and incorporating pseudo-events as priors for temporal content distribution in video moment retrieval (VMR) models. The motivation behind this study arises from the limitations of using LLMs as decoders for generating discrete textual descriptions, which hinders their direct application to continuous outputs like salience scores and inter-frame embeddings that capture inter-frame relations. To overcome these limitations, we propose utilizing LLM encoders instead of decoders. Through a feasibility study, we demonstrate that LLM encoders effectively refine inter-concept relations in multimodal embeddings, even without being trained on textual embeddings. We also show that the refinement capability of LLM encoders can be transferred to other embeddings, such as BLIP and T5, as long as these embeddings exhibit similar inter-concept similarity patterns to CLIP embeddings. We present a general framework for integrating LLM encoders into existing VMR architectures, specifically within the fusion module. Through experimental validation, we demonstrate the effectiveness of our proposed methods by achieving state-of-the-art performance in VMR. The source code can be accessed at https://github.com/fletcherjiang/LLMEPET.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10191v1">LLMs for clinical risk prediction</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      This study compares the efficacy of GPT-4 and clinalytix Medical AI in predicting the clinical risk of delirium development. Findings indicate that GPT-4 exhibited significant deficiencies in identifying positive cases and struggled to provide reliable probability estimates for delirium risk, while clinalytix Medical AI demonstrated superior accuracy. A thorough analysis of the large language model's (LLM) outputs elucidated potential causes for these discrepancies, consistent with limitations reported in extant literature. These results underscore the challenges LLMs face in accurately diagnosing conditions and interpreting complex clinical data. While LLMs hold substantial potential in healthcare, they are currently unsuitable for independent clinical decision-making. Instead, they should be employed in assistive roles, complementing clinical expertise. Continued human oversight remains essential to ensure optimal outcomes for both patients and healthcare providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10081v1">Messy Code Makes Managing ML Pipelines Difficult? Just Let LLMs Rewrite the Code!</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      Machine learning (ML) applications that learn from data are increasingly used to automate impactful decisions. Unfortunately, these applications often fall short of adequately managing critical data and complying with upcoming regulations. A technical reason for the persistence of these issues is that the data pipelines in common ML libraries and cloud services lack fundamental declarative, data-centric abstractions. Recent research has shown how such abstractions enable techniques like provenance tracking and automatic inspection to help manage ML pipelines. Unfortunately, these approaches lack adoption in the real world because they require clean ML pipeline code written with declarative APIs, instead of the messy imperative Python code that data scientists typically write for data preparation. We argue that it is unrealistic to expect data scientists to change their established development practices. Instead, we propose to circumvent this "code abstraction gap" by leveraging the code generation capabilities of large language models (LLMs). Our idea is to rewrite messy data science code to a custom-tailored declarative pipeline abstraction, which we implement as a proof-of-concept in our prototype Lester. We detail its application for a challenging compliance management example involving "incremental view maintenance" of deployed ML pipelines. The code rewrites for our running example show the potential of LLMs to make messy data science code declarative, e.g., by identifying hand-coded joins in Python and turning them into joins on dataframes, or by generating declarative feature encoders from NumPy code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10064v1">MindGuard: Towards Accessible and Sitgma-free Mental Health First Aid via Edge LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      Mental health disorders are among the most prevalent diseases worldwide, affecting nearly one in four people. Despite their widespread impact, the intervention rate remains below 25%, largely due to the significant cooperation required from patients for both diagnosis and intervention. The core issue behind this low treatment rate is stigma, which discourages over half of those affected from seeking help. This paper presents MindGuard, an accessible, stigma-free, and professional mobile mental healthcare system designed to provide mental health first aid. The heart of MindGuard is an innovative edge LLM, equipped with professional mental health knowledge, that seamlessly integrates objective mobile sensor data with subjective Ecological Momentary Assessment records to deliver personalized screening and intervention conversations. We conduct a broad evaluation of MindGuard using open datasets spanning four years and real-world deployment across various mobile devices involving 20 subjects for two weeks. Remarkably, MindGuard achieves results comparable to GPT-4 and outperforms its counterpart with more than 10 times the model size. We believe that MindGuard paves the way for mobile LLM applications, potentially revolutionizing mental healthcare practices by substituting self-reporting and intervention conversations with passive, integrated monitoring within daily life, thus ensuring accessible and stigma-free mental health support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10058v2">Learning to Refuse: Towards Mitigating Privacy Risks in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable capabilities in understanding and generating natural language. However, these models can inadvertently memorize private information, posing significant privacy risks. This study addresses the challenge of enabling LLMs to protect specific individuals' private data without the need for complete retraining. We propose \return, a Real-world pErsonal daTa UnleaRNing dataset, comprising 2,492 individuals from Wikipedia with associated QA pairs, to evaluate machine unlearning (MU) methods for protecting personal data in a realistic scenario. Additionally, we introduce the Name-Aware Unlearning Framework (NAUF) for Privacy Protection, which enables the model to learn which individuals' information should be protected without affecting its ability to answer questions related to other unrelated individuals. Our extensive experiments demonstrate that NAUF achieves a state-of-the-art average unlearning score, surpassing the best baseline method by 5.65 points, effectively protecting target individuals' personal data while maintaining the model's general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11764v2">ChatGPT Based Data Augmentation for Improved Parameter-Efficient Debiasing of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-16
      | 💬 To Appear in the Proceedings of the 1st Conference on Language Modeling (COLM) 2024
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs), while powerful, exhibit harmful social biases. Debiasing is often challenging due to computational costs, data constraints, and potential degradation of multi-task language capabilities. This work introduces a novel approach utilizing ChatGPT to generate synthetic training data, aiming to enhance the debiasing of LLMs. We propose two strategies: Targeted Prompting, which provides effective debiasing for known biases but necessitates prior specification of bias in question; and General Prompting, which, while slightly less effective, offers debiasing across various categories. We leverage resource-efficient LLM debiasing using adapter tuning and compare the effectiveness of our synthetic data to existing debiasing datasets. Our results reveal that: (1) ChatGPT can efficiently produce high-quality training data for debiasing other LLMs; (2) data produced via our approach surpasses existing datasets in debiasing performance while also preserving internal knowledge of a pre-trained LLM; and (3) synthetic data exhibits generalizability across categories, effectively mitigating various biases, including intersectional ones. These findings underscore the potential of synthetic data in advancing the fairness of LLMs with minimal retraining cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10825v2">Large Language Model (LLM) for Telecommunications: A Comprehensive Survey on Principles, Key Techniques, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have received considerable attention recently due to their outstanding comprehension and reasoning capabilities, leading to great progress in many fields. The advancement of LLM techniques also offers promising opportunities to automate many tasks in the telecommunication (telecom) field. After pre-training and fine-tuning, LLMs can perform diverse downstream tasks based on human instructions, paving the way to artificial general intelligence (AGI)-enabled 6G. Given the great potential of LLM technologies, this work aims to provide a comprehensive overview of LLM-enabled telecom networks. In particular, we first present LLM fundamentals, including model architecture, pre-training, fine-tuning, inference and utilization, model evaluation, and telecom deployment. Then, we introduce LLM-enabled key techniques and telecom applications in terms of generation, classification, optimization, and prediction problems. Specifically, the LLM-enabled generation applications include telecom domain knowledge, code, and network configuration generation. After that, the LLM-based classification applications involve network security, text, image, and traffic classification problems. Moreover, multiple LLM-enabled optimization techniques are introduced, such as automated reward function design for reinforcement learning and verbal reinforcement learning. Furthermore, for LLM-aided prediction problems, we discussed time-series prediction models and multi-modality prediction problems for telecom. Finally, we highlight the challenges and identify the future directions of LLM-enabled telecom networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09989v1">Comprehensive Study on Sentiment Analysis: From Rule-based to modern LLM based system</a></div>
    <div class="paper-meta">
      📅 2024-09-16
      | 💬 2 Images
    </div>
    <details class="paper-abstract">
      This paper provides a comprehensive survey of sentiment analysis within the context of artificial intelligence (AI) and large language models (LLMs). Sentiment analysis, a critical aspect of natural language processing (NLP), has evolved significantly from traditional rule-based methods to advanced deep learning techniques. This study examines the historical development of sentiment analysis, highlighting the transition from lexicon-based and pattern-based approaches to more sophisticated machine learning and deep learning models. Key challenges are discussed, including handling bilingual texts, detecting sarcasm, and addressing biases. The paper reviews state-of-the-art approaches, identifies emerging trends, and outlines future research directions to advance the field. By synthesizing current methodologies and exploring future opportunities, this survey aims to understand sentiment analysis in the AI and LLM context thoroughly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09916v1">SFR-RAG: Towards Contextually Faithful LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-16
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG), a paradigm that integrates external contextual information with large language models (LLMs) to enhance factual accuracy and relevance, has emerged as a pivotal area in generative AI. The LLMs used in RAG applications are required to faithfully and completely comprehend the provided context and users' questions, avoid hallucination, handle unanswerable, counterfactual or otherwise low-quality and irrelevant contexts, perform complex multi-hop reasoning and produce reliable citations. In this paper, we introduce SFR-RAG, a small LLM that is instruction-tuned with an emphasis on context-grounded generation and hallucination minimization. We also present ContextualBench, a new evaluation framework compiling multiple popular and diverse RAG benchmarks, such as HotpotQA and TriviaQA, with consistent RAG settings to ensure reproducibility and consistency in model assessments. Experimental results demonstrate that our SFR-RAG-9B model outperforms leading baselines such as Command-R+ (104B) and GPT-4o, achieving state-of-the-art results in 3 out of 7 benchmarks in ContextualBench with significantly fewer parameters. The model is also shown to be resilient to alteration in the contextual information and behave appropriately when relevant context is removed. Additionally, the SFR-RAG model maintains competitive performance in general instruction-following tasks and function-calling capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.12169v5">Rail-only: A Low-Cost High-Performance Network for Training LLMs with Trillion Parameters</a></div>
    <div class="paper-meta">
      📅 2024-09-15
    </div>
    <details class="paper-abstract">
      This paper presents a low-cost network architecture for training large language models (LLMs) at hyperscale. We study the optimal parallelization strategy of LLMs and propose a novel datacenter network design tailored to LLM's unique communication pattern. We show that LLM training generates sparse communication patterns in the network and, therefore, does not require any-to-any full-bisection network to complete efficiently. As a result, our design eliminates the spine layer in traditional GPU clusters. We name this design a Rail-only network and demonstrate that it achieves the same training performance while reducing the network cost by 38% to 77% and network power consumption by 37% to 75% compared to a conventional GPU datacenter. Our architecture also supports Mixture-of-Expert (MoE) models with all-to-all communication through forwarding, with only 8.2% to 11.2% completion time overhead for all-to-all traffic. We study the failure robustness of Rail-only networks and provide insights into the performance impact of different network and training parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13757v1">Efficient Hybrid Inference for LLMs: Reward-Based Token Modelling with Selective Cloud Assistance</a></div>
    <div class="paper-meta">
      📅 2024-09-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known for their exceptional performance across a range of natural language processing tasks, but their deployment comes at a high computational and financial cost. On the other hand, smaller language models (SLMs), which can be deployed on lower-cost edge devices, struggle to match the performance of their larger counterparts. This paper presents a novel hybrid inference approach that leverages the strengths of both model types while minimizing reliance on costly cloud-based LLMs. Unlike existing methods that route entire queries to either an SLM or a cloud LLM, our approach introduces a reward-based mechanism to dynamically determine the involvement of the cloud LLM during token generation. Specifically, each token predicted by the SLM is evaluated against a reward score, and only when this score falls below a certain threshold is the cloud LLM consulted for assistance in the next token prediction. This method not only reduces the traffic to the cloud LLM, thereby lowering costs, but also allows for flexible control over response quality depending on the reward score threshold. Experimental results demonstrate that our approach significantly reduces cloud LLM usage with minimal impact on overall response quality, offering a cost-effective solution for deploying high-performance language models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08234v2">LLM Honeypot: Leveraging Large Language Models as Advanced Interactive Honeypot Systems</a></div>
    <div class="paper-meta">
      📅 2024-09-15
      | 💬 6 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The rapid evolution of cyber threats necessitates innovative solutions for detecting and analyzing malicious activity. Honeypots, which are decoy systems designed to lure and interact with attackers, have emerged as a critical component in cybersecurity. In this paper, we present a novel approach to creating realistic and interactive honeypot systems using Large Language Models (LLMs). By fine-tuning a pre-trained open-source language model on a diverse dataset of attacker-generated commands and responses, we developed a honeypot capable of sophisticated engagement with attackers. Our methodology involved several key steps: data collection and processing, prompt engineering, model selection, and supervised fine-tuning to optimize the model's performance. Evaluation through similarity metrics and live deployment demonstrated that our approach effectively generates accurate and informative responses. The results highlight the potential of LLMs to revolutionize honeypot technology, providing cybersecurity professionals with a powerful tool to detect and analyze malicious activity, thereby enhancing overall security infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09741v1">Benchmarking LLMs in Political Content Text-Annotation: Proof-of-Concept with Toxicity and Incivility Data</a></div>
    <div class="paper-meta">
      📅 2024-09-15
      | 💬 Paper prepared for delivery at the 8th Monash-Warwick-Zurich Text-as-Data Workshop, September 16-17, 2024: 11 pages, 3 tables, 3 figures
    </div>
    <details class="paper-abstract">
      This article benchmarked the ability of OpenAI's GPTs and a number of open-source LLMs to perform annotation tasks on political content. We used a novel protest event dataset comprising more than three million digital interactions and created a gold standard that includes ground-truth labels annotated by human coders about toxicity and incivility on social media. We included in our benchmark Google's Perspective algorithm, which, along with GPTs, was employed throughout their respective APIs while the open-source LLMs were deployed locally. The findings show that Perspective API using a laxer threshold, GPT-4o, and Nous Hermes 2 Mixtral outperform other LLM's zero-shot classification annotations. In addition, Nous Hermes 2 and Mistral OpenOrca, with a smaller number of parameters, are able to perform the task with high performance, being attractive options that could offer good trade-offs between performance, implementing costs and computing time. Ancillary findings using experiments setting different temperature levels show that although GPTs tend to show not only excellent computing time but also overall good levels of reliability, only open-source LLMs ensure full reproducibility in the annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09704v1">AlpaPICO: Extraction of PICO Frames from Clinical Trial Documents Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-15
      | 💬 Accepted at Methods
    </div>
    <details class="paper-abstract">
      In recent years, there has been a surge in the publication of clinical trial reports, making it challenging to conduct systematic reviews. Automatically extracting Population, Intervention, Comparator, and Outcome (PICO) from clinical trial studies can alleviate the traditionally time-consuming process of manually scrutinizing systematic reviews. Existing approaches of PICO frame extraction involves supervised approach that relies on the existence of manually annotated data points in the form of BIO label tagging. Recent approaches, such as In-Context Learning (ICL), which has been shown to be effective for a number of downstream NLP tasks, require the use of labeled examples. In this work, we adopt ICL strategy by employing the pretrained knowledge of Large Language Models (LLMs), gathered during the pretraining phase of an LLM, to automatically extract the PICO-related terminologies from clinical trial documents in unsupervised set up to bypass the availability of large number of annotated data instances. Additionally, to showcase the highest effectiveness of LLM in oracle scenario where large number of annotated samples are available, we adopt the instruction tuning strategy by employing Low Rank Adaptation (LORA) to conduct the training of gigantic model in low resource environment for the PICO frame extraction task. Our empirical results show that our proposed ICL-based framework produces comparable results on all the version of EBM-NLP datasets and the proposed instruction tuned version of our framework produces state-of-the-art results on all the different EBM-NLP datasets. Our project is available at \url{https://github.com/shrimonmuke0202/AlpaPICO.git}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08403v5">LLMs and the Human Condition</a></div>
    <div class="paper-meta">
      📅 2024-09-15
      | 💬 Significant edits mainly to give the paper a single purpose - removed discussion of the mechanism - but just generally tighter
    </div>
    <details class="paper-abstract">
      Theory based AI research has had a hard time recently and the aim here is to propose a model of what LLMs are actually doing when they impress us with their language skills. The model integrates three established theories of human decision-making from philosophy, sociology, and computer science. The paper starts with the collective understanding of reasoning from the early days of AI research - primarily because that model is how we humans think we think, and is the most accessible. It then describes what is commonly thought of as "reactive systems" which is the position taken by many philosophers and indeed many contemporary AI researchers. The third component to the proposed model is from sociology and, although not flattering to our modern ego, provides an explanation to a puzzle that for many years has occupied those of us working on conversational user interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09774v2">HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-15
    </div>
    <details class="paper-abstract">
      Adapting a language model into a specific domain, a.k.a `domain adaption', is a common practice when specialized knowledge, e.g. medicine, is not encapsulated in a general language model like Llama2. The challenge lies in the heterogeneity of data across the two training stages, as it varies in languages, genres, or formats. To tackle this and simplify the learning protocol, we propose to transform heterogeneous data, from the both pre-training and supervised stages, into a unified, simple input-output pair format. We validate the new protocol in the domains where proprietary LLMs like ChatGPT perform relatively poorly, such as Traditional Chinese Medicine. The developed model, HuatuoGPT-II, has shown state-of-the-art performance in Chinese medicine domain on a number of benchmarks, e.g. medical licensing exams. It even outperforms proprietary models like ChatGPT and GPT-4 in some aspects, especially in Traditional Chinese Medicine. Expert manual evaluations further validate HuatuoGPT-II's advantages over existing LLMs. Notably, HuatuoGPT-II was benchmarked in a fresh Chinese National Medical Licensing Examination where it achieved the best performance, showcasing not only its effectiveness but also its generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09661v1">ContractTinker: LLM-Empowered Vulnerability Repair for Real-World Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2024-09-15
      | 💬 4 pages, and to be accepted in ASE2024
    </div>
    <details class="paper-abstract">
      Smart contracts are susceptible to being exploited by attackers, especially when facing real-world vulnerabilities. To mitigate this risk, developers often rely on third-party audit services to identify potential vulnerabilities before project deployment. Nevertheless, repairing the identified vulnerabilities is still complex and labor-intensive, particularly for developers lacking security expertise. Moreover, existing pattern-based repair tools mostly fail to address real-world vulnerabilities due to their lack of high-level semantic understanding. To fill this gap, we propose ContractTinker, a Large Language Models (LLMs)-empowered tool for real-world vulnerability repair. The key insight is our adoption of the Chain-of-Thought approach to break down the entire generation task into sub-tasks. Additionally, to reduce hallucination, we integrate program static analysis to guide the LLM. We evaluate ContractTinker on 48 high-risk vulnerabilities. The experimental results show that among the patches generated by ContractTinker, 23 (48%) are valid patches that fix the vulnerabilities, while 10 (21%) require only minor modifications. A video of ContractTinker is available at https://youtu.be/HWFVi-YHcPE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09570v1">MindScape Study: Integrating LLM and Behavioral Sensing for Personalized AI-Driven Journaling Experiences</a></div>
    <div class="paper-meta">
      📅 2024-09-15
      | 💬 arXiv admin note: text overlap with arXiv:2404.00487
    </div>
    <details class="paper-abstract">
      Mental health concerns are prevalent among college students, highlighting the need for effective interventions that promote self-awareness and holistic well-being. MindScape pioneers a novel approach to AI-powered journaling by integrating passively collected behavioral patterns such as conversational engagement, sleep, and location with Large Language Models (LLMs). This integration creates a highly personalized and context-aware journaling experience, enhancing self-awareness and well-being by embedding behavioral intelligence into AI. We present an 8-week exploratory study with 20 college students, demonstrating the MindScape app's efficacy in enhancing positive affect (7%), reducing negative affect (11%), loneliness (6%), and anxiety and depression, with a significant week-over-week decrease in PHQ-4 scores (-0.25 coefficient), alongside improvements in mindfulness (7%) and self-reflection (6%). The study highlights the advantages of contextual AI journaling, with participants particularly appreciating the tailored prompts and insights provided by the MindScape app. Our analysis also includes a comparison of responses to AI-driven contextual versus generic prompts, participant feedback insights, and proposed strategies for leveraging contextual AI journaling to improve well-being on college campuses. By showcasing the potential of contextual AI journaling to support mental health, we provide a foundation for further investigation into the effects of contextual AI journaling on mental health and well-being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11322v5">StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows</a></div>
    <div class="paper-meta">
      📅 2024-09-14
    </div>
    <details class="paper-abstract">
      It is a notable trend to use Large Language Models (LLMs) to tackle complex tasks, e.g., tasks that require a sequence of actions and dynamic interaction with tools and external environments. In this paper, we propose StateFlow, a novel LLM-based task-solving paradigm that conceptualizes complex task-solving processes as state machines. In StateFlow, we distinguish between "process grounding" (via state and state transitions) and "sub-task solving" (through actions within a state), enhancing control and interpretability of the task-solving procedure. A state represents the status of a running process. The transitions between states are controlled by heuristic rules or decisions made by the LLM, allowing for a dynamic and adaptive progression. Upon entering a state, a series of actions is executed, involving not only calling LLMs guided by different prompts, but also the utilization of external tools as needed. Our results show that StateFlow significantly enhances LLMs' efficiency. For instance, StateFlow achieves 13% and 28% higher success rates compared to ReAct in InterCode SQL and ALFWorld benchmark, with 5x and 3x less cost respectively. We also show that StateFlow can be combined with iterative refining methods like Reflexion to further improve performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04183v3">Seeing Like an AI: How LLMs Apply (and Misapply) Wikipedia Neutrality Norms</a></div>
    <div class="paper-meta">
      📅 2024-09-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on broad corpora and then used in communities with specialized norms. Is providing LLMs with community rules enough for models to follow these norms? We evaluate LLMs' capacity to detect (Task 1) and correct (Task 2) biased Wikipedia edits according to Wikipedia's Neutral Point of View (NPOV) policy. LLMs struggled with bias detection, achieving only 64% accuracy on a balanced dataset. Models exhibited contrasting biases (some under- and others over-predicted bias), suggesting distinct priors about neutrality. LLMs performed better at generation, removing 79% of words removed by Wikipedia editors. However, LLMs made additional changes beyond Wikipedia editors' simpler neutralizations, resulting in high-recall but low-precision editing. Interestingly, crowdworkers rated AI rewrites as more neutral (70%) and fluent (61%) than Wikipedia-editor rewrites. Qualitative analysis found LLMs sometimes applied NPOV more comprehensively than Wikipedia editors but often made extraneous non-NPOV-related changes (such as grammar). LLMs may apply rules in ways that resonate with the public but diverge from community experts. While potentially effective for generation, LLMs may reduce editor agency and increase moderation workload (e.g., verifying additions). Even when rules are easy to articulate, having LLMs apply them like community members may still be difficult.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09493v1">Hacking, The Lazy Way: LLM Augmented Pentesting</a></div>
    <div class="paper-meta">
      📅 2024-09-14
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Security researchers are continually challenged by the need to stay current with rapidly evolving cybersecurity research, tools, and techniques. This constant cycle of learning, unlearning, and relearning, combined with the repetitive tasks of sifting through documentation and analyzing data, often hinders productivity and innovation. This has led to a disparity where only organizations with substantial resources can access top-tier security experts, while others rely on firms with less skilled researchers who focus primarily on compliance rather than actual security. We introduce "LLM Augmented Pentesting," demonstrated through a tool named "Pentest Copilot," to address this gap. This approach integrates Large Language Models into penetration testing workflows. Our research includes a "chain of thought" mechanism to streamline token usage and boost performance, as well as unique Retrieval Augmented Generation implementation to minimize hallucinations and keep models aligned with the latest techniques. Additionally, we propose a novel file analysis approach, enabling LLMs to understand files. Furthermore, we highlight a unique infrastructure system that supports if implemented, can support in-browser assisted penetration testing, offering a robust platform for cybersecurity professionals, These advancements mark a significant step toward bridging the gap between automated tools and human expertise, offering a powerful solution to the challenges faced by modern cybersecurity teams.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09415v1">Enhancing LLM Problem Solving with REAP: Reflection, Explicit Problem Deconstruction, and Advanced Prompting</a></div>
    <div class="paper-meta">
      📅 2024-09-14
      | 💬 524 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, yet improving their problem-solving capabilities, particularly for complex, reasoning-intensive tasks, remains a persistent challenge. This paper introduces the REAP (Reflection, Explicit Problem Deconstruction, and Advanced Prompting) method, an innovative approach within the dynamic context generation framework. REAP guides LLMs through reflection on the query, deconstructing it into manageable components, and generating relevant context to enhance the solution process. We evaluated REAP using a dataset designed to expose LLM limitations, comparing zero-shot prompting with REAP-enhanced prompts across six state-of-the-art models: OpenAI's o1-preview, o1-mini, GPT-4o, GPT-4o-mini, Google's Gemini 1.5 Pro, and Claude 3.5 Sonnet. The results demonstrate notable performance gains, with o1-mini improving by 40.97%, GPT-4o by 66.26%, and GPT-4o-mini by 112.93%. Despite the already strong baseline performance of OpenAI's o1-preview, modest gains were observed. Beyond performance improvements, REAP offers a cost-effective solution; for example, GPT-4o-mini, which is approximately 100 times cheaper than o1-preview, delivered competitive results. REAP also improves the clarity of model outputs, making it easier for humans to understand the reasoning behind the results and simplifying the process of identifying and addressing any issues. These findings demonstrate REAP's potential to greatly improve the capabilities of LLMs, providing both better performance and increased cost-efficiency across a wide range of applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09354v1">PeriGuru: A Peripheral Robotic Mobile App Operation Assistant based on GUI Image Understanding and Prompting with LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-14
    </div>
    <details class="paper-abstract">
      Smartphones have significantly enhanced our daily learning, communication, and entertainment, becoming an essential component of modern life. However, certain populations, including the elderly and individuals with disabilities, encounter challenges in utilizing smartphones, thus necessitating mobile app operation assistants, a.k.a. mobile app agent. With considerations for privacy, permissions, and cross-platform compatibility issues, we endeavor to devise and develop PeriGuru in this work, a peripheral robotic mobile app operation assistant based on GUI image understanding and prompting with Large Language Model (LLM). PeriGuru leverages a suite of computer vision techniques to analyze GUI screenshot images and employs LLM to inform action decisions, which are then executed by robotic arms. PeriGuru achieves a success rate of 81.94% on the test task set, which surpasses by more than double the method without PeriGuru's GUI image interpreting and prompting design. Our code is available on https://github.com/Z2sJ4t/PeriGuru.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09345v1">Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models</a></div>
    <div class="paper-meta">
      📅 2024-09-14
    </div>
    <details class="paper-abstract">
      Agents significantly enhance the capabilities of standalone Large Language Models (LLMs) by perceiving environments, making decisions, and executing actions. However, LLM agents still face challenges in tasks that require multiple decision-making steps. Estimating the value of actions in specific tasks is difficult when intermediate actions are neither appropriately rewarded nor penalized. In this paper, we propose leveraging a task-relevant Q-value model to guide action selection. Specifically, we first collect decision-making trajectories annotated with step-level Q values via Monte Carlo Tree Search (MCTS) and construct preference data. We then use another LLM to fit these preferences through step-level Direct Policy Optimization (DPO), which serves as the Q-value model. During inference, at each decision-making step, LLM agents select the action with the highest Q value before interacting with the environment. We apply our method to various open-source and API-based LLM agents, demonstrating that Q-value models significantly improve their performance. Notably, the performance of the agent built with Phi-3-mini-4k-instruct improved by 103% on WebShop and 75% on HotPotQA when enhanced with Q-value models, even surpassing GPT-4o-mini. Additionally, Q-value models offer several advantages, such as generalization to different LLM agents and seamless integration with existing prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16073v3">Combining Fine-Tuning and LLM-based Agents for Intuitive Smart Contract Auditing with Justifications</a></div>
    <div class="paper-meta">
      📅 2024-09-14
      | 💬 Accepted for the 47th International Conference on Software Engineering (ICSE 2025)
    </div>
    <details class="paper-abstract">
      Smart contracts are decentralized applications built atop blockchains like Ethereum. Recent research has shown that large language models (LLMs) have potential in auditing smart contracts, but the state-of-the-art indicates that even GPT-4 can achieve only 30% precision (when both decision and justification are correct). This is likely because off-the-shelf LLMs were primarily pre-trained on a general text/code corpus and not fine-tuned on the specific domain of Solidity smart contract auditing. In this paper, we propose iAudit, a general framework that combines fine-tuning and LLM-based agents for intuitive smart contract auditing with justifications. Specifically, iAudit is inspired by the observation that expert human auditors first perceive what could be wrong and then perform a detailed analysis of the code to identify the cause. As such, iAudit employs a two-stage fine-tuning approach: it first tunes a Detector model to make decisions and then tunes a Reasoner model to generate causes of vulnerabilities. However, fine-tuning alone faces challenges in accurately identifying the optimal cause of a vulnerability. Therefore, we introduce two LLM-based agents, the Ranker and Critic, to iteratively select and debate the most suitable cause of vulnerability based on the output of the fine-tuned Reasoner model. To evaluate iAudit, we collected a balanced dataset with 1,734 positive and 1,810 negative samples to fine-tune iAudit. We then compared it with traditional fine-tuned models (CodeBERT, GraphCodeBERT, CodeT5, and UnixCoder) as well as prompt learning-based LLMs (GPT4, GPT-3.5, and CodeLlama-13b/34b). On a dataset of 263 real smart contract vulnerabilities, iAudit achieves an F1 score of 91.21% and an accuracy of 91.11%. The causes generated by iAudit achieved a consistency of about 38% compared to the ground truth causes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04585v3">Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness</a></div>
    <div class="paper-meta">
      📅 2024-09-14
    </div>
    <details class="paper-abstract">
      With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs and conduct extensive experiments on three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09271v1">Python Symbolic Execution with LLM-powered Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-14
    </div>
    <details class="paper-abstract">
      Symbolic execution is a key technology in software testing, which generates test cases by collecting symbolic path constraints and then solving constraints with SMT solvers. Symbolic execution has been proven helpful in generating high-coverage test cases, but its limitations, e.g., the difficulties in solving path constraints, prevent it from broader usage in software testing. Moreover, symbolic execution has encountered many difficulties when applied to dynamically typed languages like Python, because it is extremely challenging to translate the flexible Python grammar into rigid solvers. To overcome the main challenges of applying symbolic execution in Python, we proposed an LLM-empowered agent, LLM-Sym, that automatically calls an SMT solver, Z3, to solve execution path constraints. Based on an introductory-level symbolic execution engine, our LLM agent can extend it to supporting programs with complex data type `list'. The core contribution of LLM-Sym is translating complex Python path constraints into Z3 code. To enable accurate path-to-Z3 translation, we design a multiple-step code generation pipeline including type inference, retrieval and self-refine. Our experiments demonstrate that LLM-Sym is capable of solving path constraints on Leetcode problems with complicated control flows and list data structures, which is impossible for the backbone symbolic execution engine. Our approach paves the way for the combination of the generation ability of LLMs with the reasoning ability of symbolic solvers, and opens up new opportunities in LLM-augmented test case generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00761v3">Tamper-Resistant Safeguards for Open-Weight LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-14
      | 💬 Website: https://www.tamper-resistant-safeguards.com
    </div>
    <details class="paper-abstract">
      Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01811v1">Evaluating Cultural Awareness of LLMs for Yoruba, Malayalam, and English</a></div>
    <div class="paper-meta">
      📅 2024-09-14
      | 💬 19 pages, 10 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Although LLMs have been extremely effective in a large number of complex tasks, their understanding and functionality for regional languages and cultures are not well studied. In this paper, we explore the ability of various LLMs to comprehend the cultural aspects of two regional languages: Malayalam (state of Kerala, India) and Yoruba (West Africa). Using Hofstede's six cultural dimensions: Power Distance (PDI), Individualism (IDV), Motivation towards Achievement and Success (MAS), Uncertainty Avoidance (UAV), Long Term Orientation (LTO), and Indulgence (IVR), we quantify the cultural awareness of LLM-based responses. We demonstrate that although LLMs show a high cultural similarity for English, they fail to capture the cultural nuances across these 6 metrics for Malayalam and Yoruba. We also highlight the need for large-scale regional language LLM training with culturally enriched datasets. This will have huge implications for enhancing the user experience of chat-based LLMs and also improving the validity of large-scale LLM agent-based market research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09253v1">Unleash LLMs Potential for Recommendation by Coordinating Twin-Tower Dynamic Semantic Token Generator</a></div>
    <div class="paper-meta">
      📅 2024-09-14
    </div>
    <details class="paper-abstract">
      Owing to the unprecedented capability in semantic understanding and logical reasoning, the pre-trained large language models (LLMs) have shown fantastic potential in developing the next-generation recommender systems (RSs). However, the static index paradigm adopted by current methods greatly restricts the utilization of LLMs capacity for recommendation, leading to not only the insufficient alignment between semantic and collaborative knowledge, but also the neglect of high-order user-item interaction patterns. In this paper, we propose Twin-Tower Dynamic Semantic Recommender (TTDS), the first generative RS which adopts dynamic semantic index paradigm, targeting at resolving the above problems simultaneously. To be more specific, we for the first time contrive a dynamic knowledge fusion framework which integrates a twin-tower semantic token generator into the LLM-based recommender, hierarchically allocating meaningful semantic index for items and users, and accordingly predicting the semantic index of target item. Furthermore, a dual-modality variational auto-encoder is proposed to facilitate multi-grained alignment between semantic and collaborative knowledge. Eventually, a series of novel tuning tasks specially customized for capturing high-order user-item interaction patterns are proposed to take advantages of user historical behavior. Extensive experiments across three public datasets demonstrate the superiority of the proposed methodology in developing LLM-based generative RSs. The proposed TTDS recommender achieves an average improvement of 19.41% in Hit-Rate and 20.84% in NDCG metric, compared with the leading baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04927v2">LLM-based speaker diarization correction: A generalizable approach</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      Speaker diarization is necessary for interpreting conversations transcribed using automated speech recognition (ASR) tools. Despite significant developments in diarization methods, diarization accuracy remains an issue. Here, we investigate the use of large language models (LLMs) for diarization correction as a post-processing step. LLMs were fine-tuned using the Fisher corpus, a large dataset of transcribed conversations. The ability of the models to improve diarization accuracy in a holdout dataset from the Fisher corpus as well as an independent dataset was measured. We report that fine-tuned LLMs can markedly improve diarization accuracy. However, model performance is constrained to transcripts produced using the same ASR tool as the transcripts used for fine-tuning, limiting generalizability. To address this constraint, an ensemble model was developed by combining weights from three separate models, each fine-tuned using transcripts from a different ASR tool. The ensemble model demonstrated better overall performance than each of the ASR-specific models, suggesting that a generalizable and ASR-agnostic approach may be achievable. We have made the weights of these models publicly available on HuggingFace at https://huggingface.co/bklynhlth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.09051v2">Canvil: Designerly Adaptation for LLM-Powered User Experiences</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      Advancements in large language models (LLMs) are sparking a proliferation of LLM-powered user experiences (UX). In product teams, designers often craft UX to meet user needs, but it is unclear how they engage with LLMs as a novel design material. Through a formative study with 12 designers, we find that designers seek a translational mechanism that enables design requirements to shape and be shaped by LLM behavior, motivating a need for designerly adaptation to facilitate this translation. We then built Canvil, a Figma widget that operationalizes designerly adaptation. We used Canvil as a technology probe in a group-based design study (6 groups, N=17), finding that designers constructively iterated on both adaptation approaches and interface designs to enhance end-user interaction with LLMs. Furthermore, designers identified promising collaborative workflows for designerly adaptation. Our work opens new avenues for processes and tools that foreground designers' user-centered expertise in LLM-powered applications. Canvil is available for public use at https://www.figma.com/community/widget/1277396720888327660.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09135v1">Multimodal Fusion with LLMs for Engagement Prediction in Natural Conversation</a></div>
    <div class="paper-meta">
      📅 2024-09-13
      | 💬 22 pages, first three authors equal contribution
    </div>
    <details class="paper-abstract">
      Over the past decade, wearable computing devices (``smart glasses'') have undergone remarkable advancements in sensor technology, design, and processing power, ushering in a new era of opportunity for high-density human behavior data. Equipped with wearable cameras, these glasses offer a unique opportunity to analyze non-verbal behavior in natural settings as individuals interact. Our focus lies in predicting engagement in dyadic interactions by scrutinizing verbal and non-verbal cues, aiming to detect signs of disinterest or confusion. Leveraging such analyses may revolutionize our understanding of human communication, foster more effective collaboration in professional environments, provide better mental health support through empathetic virtual interactions, and enhance accessibility for those with communication barriers. In this work, we collect a dataset featuring 34 participants engaged in casual dyadic conversations, each providing self-reported engagement ratings at the end of each conversation. We introduce a novel fusion strategy using Large Language Models (LLMs) to integrate multiple behavior modalities into a ``multimodal transcript'' that can be processed by an LLM for behavioral reasoning tasks. Remarkably, this method achieves performance comparable to established fusion techniques even in its preliminary implementation, indicating strong potential for further research and optimization. This fusion method is one of the first to approach ``reasoning'' about real-world human behavior through a language model. Smart glasses provide us the ability to unobtrusively gather high-density multimodal data on human behavior, paving the way for new approaches to understanding and improving human communication with the potential for important societal benefits. The features and data collected during the studies will be made publicly available to promote further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09013v1">AI-LieDar: Examine the Trade-off Between Utility and Truthfulness in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      To be safely and successfully deployed, LLMs must simultaneously satisfy truthfulness and utility goals. Yet, often these two goals compete (e.g., an AI agent assisting a used car salesman selling a car with flaws), partly due to ambiguous or misleading user instructions. We propose AI-LieDar, a framework to study how LLM-based agents navigate scenarios with utility-truthfulness conflicts in a multi-turn interactive setting. We design a set of realistic scenarios where language agents are instructed to achieve goals that are in conflict with being truthful during a multi-turn conversation with simulated human agents. To evaluate the truthfulness at large scale, we develop a truthfulness detector inspired by psychological literature to assess the agents' responses. Our experiment demonstrates that all models are truthful less than 50% of the time, although truthfulness and goal achievement (utility) rates vary across models. We further test the steerability of LLMs towards truthfulness, finding that models follow malicious instructions to deceive, and even truth-steered models can still lie. These findings reveal the complex nature of truthfulness in LLMs and underscore the importance of further research to ensure the safe and reliable deployment of LLMs and AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08963v1">Safeguarding Decentralized Social Media: LLM Agents for Automating Community Rule Compliance</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      Ensuring content compliance with community guidelines is crucial for maintaining healthy online social environments. However, traditional human-based compliance checking struggles with scaling due to the increasing volume of user-generated content and a limited number of moderators. Recent advancements in Natural Language Understanding demonstrated by Large Language Models unlock new opportunities for automated content compliance verification. This work evaluates six AI-agents built on Open-LLMs for automated rule compliance checking in Decentralized Social Networks, a challenging environment due to heterogeneous community scopes and rules. Analyzing over 50,000 posts from hundreds of Mastodon servers, we find that AI-agents effectively detect non-compliant content, grasp linguistic subtleties, and adapt to diverse community contexts. Most agents also show high inter-rater reliability and consistency in score justification and suggestions for compliance. Human-based evaluation with domain experts confirmed the agents' reliability and usefulness, rendering them promising tools for semi-automated or human-in-the-loop content moderation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08931v1">LLM-based Weak Supervision Framework for Query Intent Classification in Video Search</a></div>
    <div class="paper-meta">
      📅 2024-09-13
      | 💬 6 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Streaming services have reshaped how we discover and engage with digital entertainment. Despite these advancements, effectively understanding the wide spectrum of user search queries continues to pose a significant challenge. An accurate query understanding system that can handle a variety of entities that represent different user intents is essential for delivering an enhanced user experience. We can build such a system by training a natural language understanding (NLU) model; however, obtaining high-quality labeled training data in this specialized domain is a substantial obstacle. Manual annotation is costly and impractical for capturing users' vast vocabulary variations. To address this, we introduce a novel approach that leverages large language models (LLMs) through weak supervision to automatically annotate a vast collection of user search queries. Using prompt engineering and a diverse set of LLM personas, we generate training data that matches human annotator expectations. By incorporating domain knowledge via Chain of Thought and In-Context Learning, our approach leverages the labeled data to train low-latency models optimized for real-time inference. Extensive evaluations demonstrated that our approach outperformed the baseline with an average relative gain of 113% in recall. Furthermore, our novel prompt engineering framework yields higher quality LLM-generated data to be used for weak supervision; we observed 47.60% improvement over baseline in agreement rate between LLM predictions and human annotations with respect to F1 score, weighted according to the distribution of occurrences of the search queries. Our persona selection routing mechanism further adds an additional 3.67% increase in weighted F1 score on top of our novel prompt engineering framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16218v2">CoverUp: Coverage-Guided LLM-Based Test Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-13
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Testing is an essential part of software development. Test generation tools attempt to automate the otherwise labor-intensive task of test creation, but generating high-coverage tests remains a challenge. This paper proposes CoverUp, a novel approach to driving the generation of high-coverage Python regression tests. CoverUp iteratively improves test coverage, interleaving coverage analysis with dialogs with the LLM that steer it to refine tests so that they increase coverage of lines and branches. We evaluate our prototype CoverUp implementation across a benchmark of challenging code derived from open-source Python projects, and show that CoverUp substantially improves on the state of the art. Compared to CodaMosa, a hybrid search/LLM-based test generator, CoverUp achieves a per-module median line+branch coverage of 80% (vs. 47%). Compared to MuTAP, a mutation/LLM-based test generator, CoverUp achieves an overall line+branch coverage of 90% (vs. 77%). We show that CoverUp's iterative, coverage-guided approach is crucial to its effectiveness, contributing to nearly 40% of its successes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08813v1">Your Weak LLM is Secretly a Strong Teacher for Alignment</a></div>
    <div class="paper-meta">
      📅 2024-09-13
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      The burgeoning capabilities of large language models (LLMs) have underscored the need for alignment to ensure these models act in accordance with human values and intentions. Existing alignment frameworks present constraints either in the form of expensive human effort or high computational costs. This paper explores a promising middle ground, where we employ a weak LLM that is significantly less resource-intensive than top-tier models, yet offers more automation than purely human feedback. We present a systematic study to evaluate and understand weak LLM's ability to generate feedback for alignment. Our empirical findings demonstrate that weak LLMs can provide feedback that rivals or even exceeds that of fully human-annotated data. Our study indicates a minimized impact of model size on feedback efficacy, shedding light on a scalable and sustainable alignment strategy. To deepen our understanding of alignment under weak LLM feedback, we conduct a series of qualitative and quantitative analyses, offering novel insights into the quality discrepancies between human feedback vs. weak LLM feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04167v5">Chinese Tiny LLM: Pretraining a Chinese-Centric Large Language Model</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      In this study, we introduce CT-LLM, a 2B large language model (LLM) that illustrates a pivotal shift towards prioritizing the Chinese language in developing LLMs. Uniquely initiated from scratch, CT-LLM diverges from the conventional methodology by primarily incorporating Chinese textual data, utilizing an extensive corpus of 1,200 billion tokens, including 800 billion Chinese tokens, 300 billion English tokens, and 100 billion code tokens. This strategic composition facilitates the model's exceptional proficiency in understanding and processing Chinese, a capability further enhanced through alignment techniques. Demonstrating remarkable performance on the CHC-Bench, CT-LLM excels in Chinese language tasks, and showcases its adeptness in English through SFT. This research challenges the prevailing paradigm of training LLMs predominantly on English corpora and then adapting them to other languages, broadening the horizons for LLM training methodologies. By open-sourcing the full process of training a Chinese LLM, including a detailed data processing procedure with the obtained Massive Appropriate Pretraining Chinese Corpus (MAP-CC), a well-chosen multidisciplinary Chinese Hard Case Benchmark (CHC-Bench), and the 2B-size Chinese Tiny LLM (CT-LLM), we aim to foster further exploration and innovation in both academia and industry, paving the way for more inclusive and versatile language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08622v1">Policy Prototyping for LLMs: Pluralistic Alignment via Interactive and Collaborative Policymaking</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      Emerging efforts in AI alignment seek to broaden participation in shaping model behavior by eliciting and integrating collective input into a policy for model finetuning. While pluralistic, these processes are often linear and do not allow participating stakeholders to confirm whether potential outcomes of their contributions are indeed consistent with their intentions. Design prototyping has long advocated for rapid iteration using tight feedback loops of ideation, experimentation, and evaluation to mitigate these issues. We thus propose policy prototyping for LLMs, a new process that draws inspiration from prototyping practices to enable stakeholders to collaboratively and interactively draft LLM policies. Through learnings from a real-world LLM policymaking initiative at an industrial AI lab, we motivate our approach and characterize policy prototyping with four guiding principles. Because policy prototyping emphasizes a contrasting set of priorities compared to previous approaches, we envision our approach to be a valuable addition to the methodological repertoire for pluralistic alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08597v1">LA-RAG:Enhancing LLM-based ASR Accuracy with Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-13
      | 💬 submitted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in integrating speech information into large language models (LLMs) have significantly improved automatic speech recognition (ASR) accuracy. However, existing methods often constrained by the capabilities of the speech encoders under varied acoustic conditions, such as accents. To address this, we propose LA-RAG, a novel Retrieval-Augmented Generation (RAG) paradigm for LLM-based ASR. LA-RAG leverages fine-grained token-level speech datastores and a speech-to-speech retrieval mechanism to enhance ASR accuracy via LLM in-context learning (ICL) capabilities. Experiments on Mandarin and various Chinese dialect datasets demonstrate significant improvements in ASR accuracy compared to existing methods, validating the effectiveness of our approach, especially in handling accent variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08554v1">LLM-Powered Grapheme-to-Phoneme Conversion: Benchmark and Case Study</a></div>
    <div class="paper-meta">
      📅 2024-09-13
      | 💬 5 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Grapheme-to-phoneme (G2P) conversion is critical in speech processing, particularly for applications like speech synthesis. G2P systems must possess linguistic understanding and contextual awareness of languages with polyphone words and context-dependent phonemes. Large language models (LLMs) have recently demonstrated significant potential in various language tasks, suggesting that their phonetic knowledge could be leveraged for G2P. In this paper, we evaluate the performance of LLMs in G2P conversion and introduce prompting and post-processing methods that enhance LLM outputs without additional training or labeled data. We also present a benchmarking dataset designed to assess G2P performance on sentence-level phonetic challenges of the Persian language. Our results show that by applying the proposed methods, LLMs can outperform traditional G2P tools, even in an underrepresented language like Persian, highlighting the potential of developing LLM-aided G2P systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04811v2">h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world. Warning: This paper and related research artifacts contain offensive and potentially disturbing prompts and model-generated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07276v2">STORE: Streamlining Semantic Tokenization and Generative Recommendation with A Single LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      Traditional recommendation models often rely on unique item identifiers (IDs) to distinguish between items, which can hinder their ability to effectively leverage item content information and generalize to long-tail or cold-start items. Recently, semantic tokenization has been proposed as a promising solution that aims to tokenize each item's semantic representation into a sequence of discrete tokens. In this way, it preserves the item's semantics within these tokens and ensures that semantically similar items are represented by similar tokens. These semantic tokens have become fundamental in training generative recommendation models. However, existing generative recommendation methods typically involve multiple sub-models for embedding, quantization, and recommendation, leading to an overly complex system. In this paper, we propose to streamline the semantic tokenization and generative recommendation process with a unified framework, dubbed STORE, which leverages a single large language model (LLM) for both tasks. Specifically, we formulate semantic tokenization as a text-to-token task and generative recommendation as a token-to-token task, supplemented by a token-to-text reconstruction task and a text-to-token auxiliary task. All these tasks are framed in a generative manner and trained using a single LLM backbone. Extensive experiments have been conducted to validate the effectiveness of our STORE framework across various recommendation tasks and datasets. We will release the source code and configurations for reproducible research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06816v2">LLM-Enhanced Software Patch Localization</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      Open source software (OSS) is integral to modern product development, and any vulnerability within it potentially compromises numerous products. While developers strive to apply security patches, pinpointing these patches among extensive OSS updates remains a challenge. Security patch localization (SPL) recommendation methods are leading approaches to address this. However, existing SPL models often falter when a commit lacks a clear association with its corresponding CVE, and do not consider a scenario that a vulnerability has multiple patches proposed over time before it has been fully resolved. To address these challenges, we introduce LLM-SPL, a recommendation-based SPL approach that leverages the capabilities of the Large Language Model (LLM) to locate the security patch commit for a given CVE. More specifically, we propose a joint learning framework, in which the outputs of LLM serves as additional features to aid our recommendation model in prioritizing security patches. Our evaluation on a dataset of 1,915 CVEs associated with 2,461 patches demonstrates that LLM-SPL excels in ranking patch commits, surpassing the state-of-the-art method in terms of Recall, while significantly reducing manual effort. Notably, for vulnerabilities requiring multiple patches, LLM-SPL significantly improves Recall by 22.83\%, NDCG by 19.41\%, and reduces manual effort by over 25\% when checking up to the top 10 rankings. The dataset and source code are available at \url{https://anonymous.4open.science/r/LLM-SPL-91F8}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08493v1">Intelligent LiDAR Navigation: Leveraging External Information and Semantic Maps with LLM as Copilot</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      Traditional robot navigation systems primarily utilize occupancy grid maps and laser-based sensing technologies, as demonstrated by the popular move_base package in ROS. Unlike robots, humans navigate not only through spatial awareness and physical distances but also by integrating external information, such as elevator maintenance updates from public notification boards and experiential knowledge, like the need for special access through certain doors. With the development of Large Language Models (LLMs), which posses text understanding and intelligence close to human performance, there is now an opportunity to infuse robot navigation systems with a level of understanding akin to human cognition. In this study, we propose using osmAG (Area Graph in OpensStreetMap textual format), an innovative semantic topometric hierarchical map representation, to bridge the gap between the capabilities of ROS move_base and the contextual understanding offered by LLMs. Our methodology employs LLMs as actual copilot in robot navigation, enabling the integration of a broader range of informational inputs while maintaining the robustness of traditional robotic navigation systems. Our code, demo, map, experiment results can be accessed at https://github.com/xiexiexiaoxiexie/Intelligent-LiDAR-Navigation-LLM-as-Copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04667v2">LLM Stability: A detailed analysis with some surprises</a></div>
    <div class="paper-meta">
      📅 2024-09-12
    </div>
    <details class="paper-abstract">
      LLM (large language model) practitioners commonly notice that outputs can vary for the same inputs, but we have been unable to find work that evaluates LLM stability as the main objective. In our study of 6 deterministically configured LLMs across 8 common tasks with 5 identical runs, we see accuracy variations up to 10\%. In addition, no LLM consistently delivers repeatable accuracy across all tasks. We also show examples of variation that are not normally distributed and compare configurations with zero-shot/few-shot prompting and fine-tuned examples. To better quantify what is going on, we introduce metrics focused on stability: TARr@N for the total agreement rate at N runs over raw output, and TARa@N for total agreement over parsed-out answers. We suggest that stability metrics be integrated into leader boards and research results going forward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08147v1">LLM-POTUS Score: A Framework of Analyzing Presidential Debates with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-09-12
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated remarkable capabilities in natural language processing, yet their application to political discourse analysis remains underexplored. This paper introduces a novel approach to evaluating presidential debate performances using LLMs, addressing the longstanding challenge of objectively assessing debate outcomes. We propose a framework that analyzes candidates' "Policies, Persona, and Perspective" (3P) and how they resonate with the "Interests, Ideologies, and Identity" (3I) of four key audience groups: voters, businesses, donors, and politicians. Our method employs large language models to generate the LLM-POTUS Score, a quantitative measure of debate performance based on the alignment between 3P and 3I. We apply this framework to analyze transcripts from recent U.S. presidential debates, demonstrating its ability to provide nuanced, multi-dimensional assessments of candidate performances. Our results reveal insights into the effectiveness of different debating strategies and their impact on various audience segments. This study not only offers a new tool for political analysis but also explores the potential and limitations of using LLMs as impartial judges in complex social contexts. In addition, this framework provides individual citizens with an independent tool to evaluate presidential debate performances, which enhances democratic engagement and reduces reliance on potentially biased media interpretations and institutional influence, thereby strengthening the foundation of informed civic participation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17166v1">ScriptSmith: A Unified LLM Framework for Enhancing IT Operations via Automated Bash Script Generation, Assessment, and Refinement</a></div>
    <div class="paper-meta">
      📅 2024-09-12
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      In the rapidly evolving landscape of site reliability engineering (SRE), the demand for efficient and effective solutions to manage and resolve issues in site and cloud applications is paramount. This paper presents an innovative approach to action automation using large language models (LLMs) for script generation, assessment, and refinement. By leveraging the capabilities of LLMs, we aim to significantly reduce the human effort involved in writing and debugging scripts, thereby enhancing the productivity of SRE teams. Our experiments focus on Bash scripts, a commonly used tool in SRE, and involve the CodeSift dataset of 100 tasks and the InterCode dataset of 153 tasks. The results show that LLMs can automatically assess and refine scripts efficiently, reducing the need for script validation in an execution environment. Results demonstrate that the framework shows an overall improvement of 7-10% in script generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07871v1">Objection Overruled! Lay People can Distinguish Large Language Models from Lawyers, but still Favour Advice from an LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-12
      | 💬 13 pages, 6 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are seemingly infiltrating every domain, and the legal context is no exception. In this paper, we present the results of three experiments (total N=288) that investigated lay people's willingness to act upon, and their ability to discriminate between, LLM- and lawyer-generated legal advice. In Experiment 1, participants judged their willingness to act on legal advice when the source of the advice was either known or unknown. When the advice source was unknown, participants indicated that they were significantly more willing to act on the LLM-generated advice. This result was replicated in Experiment 2. Intriguingly, despite participants indicating higher willingness to act on LLM-generated advice in Experiments 1 and 2, participants discriminated between the LLM- and lawyer-generated texts significantly above chance-level in Experiment 3. Lastly, we discuss potential explanations and risks of our findings, limitations and future work, and the importance of language complexity and real-world comparability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07829v1">Enabling Cost-Effective UI Automation Testing with Retrieval-Based LLMs: A Case Study in WeChat</a></div>
    <div class="paper-meta">
      📅 2024-09-12
    </div>
    <details class="paper-abstract">
      UI automation tests play a crucial role in ensuring the quality of mobile applications. Despite the growing popularity of machine learning techniques to generate these tests, they still face several challenges, such as the mismatch of UI elements. The recent advances in Large Language Models (LLMs) have addressed these issues by leveraging their semantic understanding capabilities. However, a significant gap remains in applying these models to industrial-level app testing, particularly in terms of cost optimization and knowledge limitation. To address this, we introduce CAT to create cost-effective UI automation tests for industry apps by combining machine learning and LLMs with best practices. Given the task description, CAT employs Retrieval Augmented Generation (RAG) to source examples of industrial app usage as the few-shot learning context, assisting LLMs in generating the specific sequence of actions. CAT then employs machine learning techniques, with LLMs serving as a complementary optimizer, to map the target element on the UI screen. Our evaluations on the WeChat testing dataset demonstrate the CAT's performance and cost-effectiveness, achieving 90% UI automation with $0.34 cost, outperforming the state-of-the-art. We have also integrated our approach into the real-world WeChat testing platform, demonstrating its usefulness in detecting 141 bugs and enhancing the developers' testing process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12093v2">LLM-enhanced Scene Graph Learning for Household Rearrangement</a></div>
    <div class="paper-meta">
      📅 2024-09-12
      | 💬 SIGGRAPH ASIA 2024 conference accepted
    </div>
    <details class="paper-abstract">
      The household rearrangement task involves spotting misplaced objects in a scene and accommodate them with proper places. It depends both on common-sense knowledge on the objective side and human user preference on the subjective side. In achieving such task, we propose to mine object functionality with user preference alignment directly from the scene itself, without relying on human intervention. To do so, we work with scene graph representation and propose LLM-enhanced scene graph learning which transforms the input scene graph into an affordance-enhanced graph (AEG) with information-enhanced nodes and newly discovered edges (relations). In AEG, the nodes corresponding to the receptacle objects are augmented with context-induced affordance which encodes what kind of carriable objects can be placed on it. New edges are discovered with newly discovered non-local relations. With AEG, we perform task planning for scene rearrangement by detecting misplaced carriables and determining a proper placement for each of them. We test our method by implementing a tiding robot in simulator and perform evaluation on a new benchmark we build. Extensive evaluations demonstrate that our method achieves state-of-the-art performance on misplacement detection and the following rearrangement planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07772v1">Alignment with Preference Optimization Is All You Need for LLM Safety</a></div>
    <div class="paper-meta">
      📅 2024-09-12
    </div>
    <details class="paper-abstract">
      We demonstrate that preference optimization methods can effectively enhance LLM safety. Applying various alignment techniques to the Falcon 11B model using safety datasets, we achieve a significant boost in global safety score (from $57.64\%$ to $99.90\%$) as measured by LlamaGuard 3 8B, competing with state-of-the-art models. On toxicity benchmarks, average scores in adversarial settings dropped from over $0.6$ to less than $0.07$. However, this safety improvement comes at the cost of reduced general capabilities, particularly in math, suggesting a trade-off. We identify noise contrastive alignment (Safe-NCA) as an optimal method for balancing safety and performance. Our study ultimately shows that alignment techniques can be sufficient for building safe and robust models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07604v1">Multilingual Prompts in LLM-Based Recommenders: Performance Across Languages</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in natural language processing tasks. Recommender systems traditionally use methods such as collaborative filtering and matrix factorization, as well as advanced techniques like deep learning and reinforcement learning. Although language models have been applied in recommendation, the recent trend have focused on leveraging the generative capabilities of LLMs for more personalized suggestions. While current research focuses on English due to its resource richness, this work explores the impact of non-English prompts on recommendation performance. Using OpenP5, a platform for developing and evaluating LLM-based recommendations, we expanded its English prompt templates to include Spanish and Turkish. Evaluation on three real-world datasets, namely ML1M, LastFM, and Amazon-Beauty, showed that usage of non-English prompts generally reduce performance, especially in less-resourced languages like Turkish. We also retrained an LLM-based recommender model with multilingual prompts to analyze performance variations. Retraining with multilingual prompts resulted in more balanced performance across languages, but slightly reduced English performance. This work highlights the need for diverse language support in LLM-based recommenders and suggests future research on creating evaluation datasets, using newer models and additional languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07587v1">Exploring LLMs for Malware Detection: Review, Framework Design, and Countermeasure Approaches</a></div>
    <div class="paper-meta">
      📅 2024-09-11
      | 💬 26 pages, 7 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The rising use of Large Language Models (LLMs) to create and disseminate malware poses a significant cybersecurity challenge due to their ability to generate and distribute attacks with ease. A single prompt can initiate a wide array of malicious activities. This paper addresses this critical issue through a multifaceted approach. First, we provide a comprehensive overview of LLMs and their role in malware detection from diverse sources. We examine five specific applications of LLMs: Malware honeypots, identification of text-based threats, code analysis for detecting malicious intent, trend analysis of malware, and detection of non-standard disguised malware. Our review includes a detailed analysis of the existing literature and establishes guiding principles for the secure use of LLMs. We also introduce a classification scheme to categorize the relevant literature. Second, we propose performance metrics to assess the effectiveness of LLMs in these contexts. Third, we present a risk mitigation framework designed to prevent malware by leveraging LLMs. Finally, we evaluate the performance of our proposed risk mitigation strategies against various factors and demonstrate their effectiveness in countering LLM-enabled malware. The paper concludes by suggesting future advancements and areas requiring deeper exploration in this fascinating field of artificial intelligence.
    </details>
</div>
