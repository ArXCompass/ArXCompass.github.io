# llm - 2026_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.25727v2">Auditing Privacy Risks in LLM-Enhanced Graph Neural Networks</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently advanced graph neural networks (GNNs) by enriching node representations with semantic information, giving rise to LLM-enhanced GNNs that achieve substantial performance gains. However, how such semantic enhancement affects privacy risks remains largely underexplored. To bridge this gap, we systematically audit the privacy risks of LLM-enhanced GNNs through a unified framework consisting of five stages: (1) dataset preparation, (2) victim model training, (3) privacy attack, (4) risk assessment, and (5) defense analysis. Specifically, our evaluation spans ten text-attributed graph datasets across diverse domains, six privacy attacks, 42 LLM-enhanced GNN configurations, and three more recent language-model backbones. Extensive experiments show that, despite their utility improvements, LLM-enhanced GNNs consistently exhibit greater empirical privacy vulnerability than shallow text representation baselines under the evaluated attacks across diverse models and datasets. Further analysis shows that LLM-enhanced representations exhibit more distinguishable link-, label-, and membership-related signals in the embedding space, making them more exploitable by inference attacks. Finally, we evaluate representative defenses and examine their effectiveness in mitigating these privacy risks. Overall, this work provides a systematic audit of privacy risks in LLM-enhanced GNNs and offers insights for developing more secure and trustworthy graph learning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04281v5">Reasoning Enhances Robustness to Prompt Injection in LLM-Based Consensus</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are gaining traction as a method to generate consensus statements and aggregate preferences in digital democracy experiments. Yet, participants can introduce critical vulnerabilities in LLM-based systems. Here, we examine the vulnerability and robustness of off-the-shelf consensus-generating LLMs to prompt-injection attacks, which consist of introducing strategically designed texts to amplify particular viewpoints, erase certain opinions, or divert consensus toward unrelated or irrelevant topics. Using data collected from a 2023 experiment conducted in the UK and predefining a majority-rule, we construct attack-free and adversarial variants of prompts containing public policy questions and opinion texts, classify opinion and consensus valences with a fine-tuned BERT model, and estimate LLM--human majority agreement rates. Overall, we find that default LLMs exhibit widespread vulnerability, especially when: (i) disagreement and agreement are finely balanced, (ii) under rational, instruction-like rhetorical strategies, and (iii) for attacks that shift consensus toward positions aligned with GB-unionist conservative manifestos relative to pro-independence left manifestos. A robustness pipeline combining GPT-OSS-SafeGuard injection detection, structured opinion representations, and GSPO-based reinforcement learning substantially reduces directional failures, outperforming state-of-the-art alternatives. While more effective detectors may enable more sophisticated attacks, these findings advance our understanding of both the vulnerabilities and the potential defenses of consensus-generating LLMs in digital democracy applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10058v1">Cache the Encoder Within:Compact, Reusable Memory across LLM Queries</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 17 pages, 3 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Repeated queries over shared documents incur redundant encoding, while caching model states introduces persistent storage costs. Building on CoMem's intermediate-state interface, EncBank treats a pretrained LLM's lower layers as a reusable document encoder and compactly stores their outputs for an adapted upper-layer reader. A self-distilled suffix adapter is shared across storage precisions within each backbone, without quantization-specific retraining. Across five benchmark suites on three Qwen backbones spanning different sizes and full-attention and hybrid architectures, 4-bit storage keeps each reported benchmark aggregate within one score point of native-precision EncBank. In a fixed Qwen3-8B workload, it retains 28.1% of the native-precision persistent GPU store. Separate native-precision controls yield a 1.40x selected-pack prefill speedup over same-evidence, same-adapter text replay, at a 3.12-point RULER accuracy cost. A native-precision Qwen3.8-27B configuration also passes 70 of 89 Terminal-Bench 2.1 tasks. EncBank thus combines reusable computation with compact memory, while task fidelity and end-to-end benefits remain dependent on the workload, preparation costs, and reuse frequency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10633v1">Has LLM Screening Performance Stalled in Software Engineering Systematic Reviews?</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 53 pages, four external figures available in the research artifact
    </div>
    <details class="paper-abstract">
      Screening in systematic reviews (SRs) is manual and time-consuming. Prior work has explored large language models (LLMs) for automating this step, but LLMs are evolving rapidly, so earlier performance claims may no longer accurately reflect their screening performance. We used an existing software engineering SR screening benchmark (SESR-Eval) as our data. We also power-sampled a new, smaller dataset (SESR-Eval-Mini) that allows evaluation at lower costs. Using this data, we evaluated eight new LLMs for screening performance. Additionally, we tested different prompts, analyzed LLM agreement in screening decisions and criteria, and examined the effect of refining the inclusion and exclusion criteria on screening performance. The eight new LLMs performed marginally better than the seven old ones: avg. MCC across secondary studies rose from 0.347 to 0.365. Differences between secondary studies are still bigger than between LLMs. Computing the overall screening decision from criterion-level decisions degraded screening performance only slightly. LLMs generally agree with each other in their corresponding screening decisions (mean Gwet's AC1 = 0.830), though certain inclusion and exclusion criteria showed larger disagreement than others. Refining the inclusion and exclusion criteria slightly improved recall and made decisions easier for some LLMs, but overall impacts of criteria refinement were modest. LLMs are not yet ready to replace humans in paper screening and the advantages new, more costly models bring, appear to be very limited. Agent-based approaches, prompt engineering, and further criteria refinement are three potential future research avenues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.33181v2">SeOPD: Self-Evolving LLMs via Online Policy Distillation from Self-Generated Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Recent advances in online policy self-distillation (OPSD) have demonstrated that large language models (LLMs) can improve their capabilities by leveraging external privileged information (PI), such as manual annotations or feedback from external environments. However, obtaining accurate annotations and constructing sophisticated environments often require substantial human effort and computation, limiting the scalability of OPSD. While a few recent studies have explored self-improvement without external PI, the resulting gains remain limited. In this work, we explore whether LLMs can achieve comparable self-improvement without external PI. Our key observation is that a single LLM can support multiple reasoning modes, such as deep-thinking and non-thinking modes, with deep thinking generating additional information during reasoning. Based on this observation, we propose Self-Evolving Online Policy Distillation (SeOPD), which enables LLMs to distill and internalize information generated by their own chain of thought (CoT). Specifically, it (1) generates CoT with the deep-thinking mode, (2) produces responses with the non-thinking mode, and (3) uses the generated CoT as PI to provide token-level supervision for the non-thinking response, allowing new information inferred during reasoning to guide the non-thinking mode and be internalized into the shared model parameters, thereby improving both non-thinking and deep-thinking capabilities. Extensive experiments across LLMs and tasks demonstrate the effectiveness of SeOPD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09985v1">Marrying Pricing and Advertising with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      We study a sequential pricing problem in which a seller jointly posts a price and an advertisement generated by a large language model (LLM). The seller aims to maximize revenue under an unknown product demand that depends on both decisions, while observing only whether each offer leads to a purchase. We propose an online actor-critic algorithm that combines low-rank adaptation (LoRA) of a pretrained LLM with a demand model fitted to available data. At each round, the actor generates an advertisement, and the critic estimates purchase probabilities to guide price selection. Then, the resulting feedback is used to update both the actor and the critic, with the critic's revenue estimates providing a baseline for policy gradient updates of the actor. To evaluate our approach, we develop an evaluation framework with three synthetic demand models and a demand simulator built from real-world marketplace data. Finally, we compare our algorithm with benchmarks that do not jointly optimize price selection and advertisement generation, achieving expected revenue gains over the reference policy of 5.69%, 5.18% and 55.96% under the three synthetic demand models and 5.81% under the marketplace simulator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09981v1">Sensitive-Topic Leakage Through LLM Routing Metadata: Measurement and Mitigation</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 20 pages, 5 figures, 9 tables
    </div>
    <details class="paper-abstract">
      LLM routers pick a cheap or expensive model per request by its content, and many gateways and some cloud platforms can log that choice with content logging off. We measure this privacy channel beyond token counts, accounting for noisy labels and repeated prompts. We run pre-registered studies on 1.7 million real requests (WildChat-1M, LMSYS-Chat-1M) with two cost/quality routers and a domain router, survey eleven systems' logging, and test post-processing defenses. At matched length, the shift's direction depends on category and router. For RouteLLM at the 50% operating point, harassment and self-harm requests reach the strong model 19 points less often than comparable ones on prompts unseen in exploration, medical requests (exploratory: LLM labels failed their gate) 31 points less often on distinct prompts (both post hoc), and sexual requests 10 points more often (secondary); the other router's four are negative. Twenty RouteLLM decisions separate frequent medical askers with AUC 0.71, exploratory and below the pre-registered primary endpoint's 0.75 (domain router: 0.92, an upper estimate). Per-category length-matched parity with accurate labels removes the gap on real traffic, costing at most 0.2 accuracy points on RouterBench (post hoc), where routers' gaps on sensitive subjects (13-42 points, pre-registered) exceed those of an oracle routing by realized accuracy gain (1-11, post hoc). Per-conversation stickiness, per-user budget bands, and pooled parity fail, the last as categories' shifts differ in size or sign. A post hoc exact per-user rate hides only even-prefix strong counts and forfeits most self-assessed routing value; it preserves odd-position decisions, from which a post hoc log attack reaches AUC 0.73 after 20 RouteLLM requests (exploratory).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09973v1">From Expected Harmfulness to Likelihood: A Probabilistic Reformulation of Jailbreaking LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      When the harmfulness of an LLM agent's output can be quantified, a natural jailbreaking objective is to maximize expected harmfulness over admissible input modifications. An alternative approach constructs or selects harmful target outputs and modifies the input to increase their likelihood. We establish a precise connection between these two approaches through a probabilistic reformulation. Specifically, we show that the gradient of the logarithm of expected harmfulness with respect to the input equals the expected input gradient of the model's log-likelihood under a harmfulness reweighted output distribution. This identity provides a unified interpretation of expected harmfulness and target likelihood optimization. Building on this connection, we propose OPUR, a sampling distribution designed to generate highly harmful target outputs and use the resulting samples to guide likelihood-based input optimization. Experiments demonstrate the effectiveness of the resulting method in jailbreaking LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10629v1">The Harness as the Only Mutable Surface: Compliance-Bounded Self-Evolution of LLM Agents in Credit Pipelines, with a Measured Admission Gate</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 14 pages, 5 tables. Code, configuration and per-run outputs: https://github.com/rav11l/cace-bench (v0.6.0, doi:10.5281/zenodo.23207546)
    </div>
    <details class="paper-abstract">
      Self-improving LLM agents can adapt a credit pipeline to a changed rule, but an agent that rewrites itself destroys the artefact a supervisor reviews: a named change, a recorded test, an approval. We argue that self-evolution is reviewable only if it is confined to the runtime harness (instruction text, tool-call logic and primitive composition) while model weights stay fixed, so that every adaptation is a diff with a cause and a test attached. We give a dual-loop engine built on that bound, with one admission gate that writes a hash-chained record before deployment, and we measure the gate in simulation, with a simulated agent and a seeded-search proposer rather than language models. Across three families of supervisory re-interpretation at three severities, 10 seeds each, the gated loop admitted 144 of 7,449 candidate changes, none of which worsened error on held-out history, and restored the false-positive rate to the oracle level without raising missed flags in every low- and mid-severity cell. With the gate replaced by the check an unbounded system applies (fewer errors visible in recent traces), the same loops admitted 309 harmful changes and left missed flags above 10% in 49 of 90 runs: false positives fell because the screen was loosened. Evaluated on pre-shift labels, the gate rejected every candidate, so a re-interpretation must be encoded as a rule that relabels history. Parametric and scope shifts were repaired locally, a structural one only by primitive replacement; at the highest structural severity the gate's fixed tolerance blocked the correct replacement in half the seeds. We map the mechanisms to the EU AI Act's provisions for high-risk credit scoring and note that the April 2026 US model-risk guidance excludes agentic AI from its scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.27817v4">Auditing generative audio calls for known-task audio-llm evaluation</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Speech and audio LLMs are evaluated by comparing waveform predictions with predictions from an automatic speech recognition (ASR) transcript. For fixed closed-set tasks, this conflates acoustic evidence with the need to invoke a generative audio model. We estimate incremental call value with matched selectors sharing pre-call evidence. Each policy may retain the transcript label, use a local encoder, or invoke a generative model; matched control removes generative actions but preserves pre-call evidence and development selection. On VocalSound, transcript-only accuracy is 0.296, while supervised CLAP and WavLM controls reach 0.850 and 0.854 without calls. Full selector reaches 0.925 at 12.5% calls versus 0.921 for matched No-call selector (difference 0.004; 95% CI [-0.025, 0.033]). Thus, results do not show a call gain after transcript and encoder evidence are available. Relevant quantity is incremental accuracy from allowing calls, not the waveform-transcript gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09872v1">LiveMACE: Process-Aware Evaluation of LLM Agent Capabilities in Evolving Markets</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Evaluating agents by outcomes alone can obscure the capabilities that produce them. This problem is especially pronounced in evolving environments, where outcomes reflect a closed-loop interaction between agent behavior and changing external conditions. We introduce LiveMACEBench, a process-aware benchmark that uses live financial markets as a naturally evolving testbed for persistent LLM agents. Five frontier LLMs operate along continuous trajectories under matched Tool Use, Persistent Memory, Rule Following, and Multi-Agent Collaboration configurations. We evaluate them through both realized outcomes and mechanism-specific diagnostics derived from complete decision traces. Across 30 days of live evaluation, we find a pronounced outcome-capability gap: realized returns often diverge from capability-specific measurements, and similar outcomes can arise from markedly different patterns of mechanism use. Trace-level diagnostics further expose distinct bottlenecks across capabilities, demonstrating that mechanism access, effective mechanism use, and downstream performance are not interchangeable measures of agent capability. LiveMACEBench makes this distinction measurable, turning live markets from a performance leaderboard into a diagnostic environment for agent capability
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09820v1">The Impact of Backbone Evolution on LLM-Based Relevance Assessments</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 12 pages main content
    </div>
    <details class="paper-abstract">
      LLMs are evolving rapidly, with newer models offering stronger capabilities. This suggests that in LLM-based relevance judging, more capable models will achieve higher agreement with human judgements under the same prompt. We challenge this understanding by investigating the behavior of LLM-based relevance judges under backbone evolution. Keeping the prompts fixed, we evaluate a representative single-prompt (UMBRELA) and a rubric-based prompt (EXAM) across sequential model versions of commercial (Gemini, GPT) and open-weight (Qwen, Llama) models. Overall, we find no consistent evidence that newer versions lead to better relevance judges. Crucially, similar or improved aggregate performance does not imply judgment stability: correct judgements made by an earlier version of an LLM backbone are not necessarily preserved by later versions. We investigate the potential drivers of these regressions. Our findings caution against the assumption that judging prompts designed and validated for one backbone version will perform equivalently or better when the model is updated, even within the same family.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09795v1">MIRROR: From Imitation to Internalization in LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 36 pages
    </div>
    <details class="paper-abstract">
      The demand for personalized LLMs is shifting from style imitation toward content quality. We investigate whether self-distillation can bridge this gap in existing fine-tuning paradigm. To address this limitation, we introduce MIRROR(Meta- personalization by Internalizing Reference-Revealed On-policy Reflections), a novel self-distillation framework that shifts LLM personalization from imitation toward preference internalization. First, we replace reference-token imitation with reference-revealed on-policy self-distillation, aligning the model's next-token distributions along its own generation trajectories with those of its reference-conditioned self, thereby internalizing user preferences rather than reproducing reference wording.Second, we introduce MIRROR-F, a focal plug-in that augments on-policy distributional alignment with selective supervision over informative reference tokens, thereby strengthening content generation while preserving user-specific expression. Across three personalized generation benchmarks, two model scales, and complementary reference-based and LLM-based evaluations, MIRROR and MIRROR-F achieve leading overall personalization performance and superior text quality, while exhibiting less catastrophic forgetting than SFT-based baselines on three unseen personalized generation tasks. The gains are consistent across model scales and application scenarios, translating to improved performance in LLM personalization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09793v1">Formal Runtime Verification for Tool-Using LLM Agents: An Offline Same-Benchmark Study on AgentDojo and STAC</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 14 pages, 1 figure, 13 tables. Accepted at the 8th Workshop on CPS&IoT Security and Privacy (CPSIoTSec '26), co-located with ACM CCS 2026, The Hague
    </div>
    <details class="paper-abstract">
      Guardrails for tool-using LLM agents are usually application-specific rules, which makes multi-step, data-dependent safety policies hard to specify, audit and reuse. As a declarative alternative, we evaluate metric first-order temporal logic (MFOTL), replaying the recorded trajectories that AgentDojo, STAC and R-Judge already ship through the unmodified MonPoly monitor, offline and without running an agent. On these corpora, five generic obligations flag 71.8% of STAC attack chains and 70.1% of successful AgentDojo attacks, but also fire on 29.3% of benign runs. This imprecision stems from the corpora rather than the logic: they rarely record approvals and never record timestamps, so history-dependent obligations reduce to detecting risky action types. Where the trace does carry relational context, provenance-aware policies discriminate better; that context, however, is itself attackable, and one planted line defeats a naive provenance check on 94-99% of the runs it would otherwise flag. Binding provenance to the lookup that produced it closes this evasion at no cost in detection or benign firing. Taken together, these results show that formal temporal monitoring adds value exactly when the trace exposes trustworthy history. We therefore quantify how far current benchmarks are from that point and propose a twelve-field enforcement-ready trace schema.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09778v1">Reproducible LLM Inference Benchmarking: A Sequential Isolation Protocol for Regression Testing</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Reproducible benchmarking of Large Language Model (LLM) inference is challenging because repeated measurements can vary with execution and system state. We present the Sequential Isolation Methodology, a controlled benchmarking and regression-testing protocol designed to reduce between-run measurement variance while deliberately varying workload concurrency. We evaluate three representative open-source LLMs on an NVIDIA A100 80GB GPU using vLLM 0.9.1 across six context sizes and eight concurrency levels, with five repetitions per configuration. The final protocol reduces average coefficient of variation (CV) from 15.2% in the least controlled methodology stage to 2.2% under the final protocol; using CV computed across the five repetition-level median (P50) TTFT values per configuration, 113 of 144 configurations (78.5%) achieve CV below 3%. The measurements also show a marked latency transition between 200 and 500 concurrent users on the tested stack and descriptive differences in P99 latency across the three models. We additionally provide an explicit cost break-even model with sensitivity to API pricing. The protocol is intended to provide a stable reference for reproducible comparison and regression testing rather than to predict absolute behavior under uncontrolled production traffic. Infrastructure-as-Code and benchmark scripts support replication of the experimental environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09733v1">Bridge Routing Heads: Where Multilingual Multi-hop Reasoning Lives in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Accepted at EMNLP 2026
    </div>
    <details class="paper-abstract">
      Multilingual LLMs answer the same multi-hop reasoning question across languages, but we lack a mechanistic account of whether they share an internal circuit. We identify Bridge Routing Heads (BRH) in two large multilingual LLMs through a three-stage pipeline. The resulting language-specific head sets exhibit near-complete mutual exclusivity across the five languages, with a mean Jaccard similarity of only 0.017 for Llama 3.1 70B and 0.057 for Qwen 2.5 72B, revealing language-idiosyncratic circuits. Ablating general BRH increases two-hop Negative Log-Likelihood (NLL) by 39-89x the random-head baseline, providing direct causal evidence of their role. Amplifying these heads in a failing target-language pass rescues up to 51.7% of cross-lingual failures, with no training. The two models share this dual-circuit pattern but allocate heads differently: Llama concentrates chaining in a large general pool, while Qwen leans on larger language-specific pools. Together these results show that activation-level intervention alone can recover correct answers from cross-lingual reasoning failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10622v1">WorldBench: Evaluating LLMs on Three.js Voxel World Generation</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Large language models can now write complete, interactive 3D worlds as code, but grading those worlds automatically is unreliable. Existing judges take one view of the output: a vision-language model scores a few rendered snapshots, or a language model reads the source. On worlds written by five frontier models we find that the two views disagree on 32% of required items, mostly code that no frame shows, and that fixed views miss small close-up contents. We present WorldBench, a benchmark and judge for open-ended, LLM-generated Three.js worlds. From one prompt describing a floating voxel island with ten biomes, physics, and day/night and seasonal cycles, the judge explores the running world, controlling its clock, orbiting it, and sending a navigator agent to frame each biome, and reads the code for what it sees. Neither channel is trusted on its own: a code quote counts only if it is text the source contains, and visual claims are checked against measured pixels where the property is measurable. A mutation test, in which we remove features by construction, shows that code-only judging gives full credit to four of five removed features, because their code remains in the file. Our judge cuts the points kept on removed features by a third (5.44 to 3.55 of 7.11), and what it still credits is mostly code that exists but never runs. We evaluate five frontier models: Claude Fable 5.1, GPT-6 Astra, Kimi K3, Grok 4.7 and Gemini 3.1 Pro. Code, prompt, tests and judge configuration are available at https://github.com/KrishBakshi/worldbench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09647v1">When Rank Rises as LLMs Degrade</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 NeurIPS 2026 Workshop on Continual Learning for Foundation Models and Agents (CL4FMAgents); 8 pages + appendix
    </div>
    <details class="paper-abstract">
      Post-training adapts language models in non-stationary environments. Practitioners monitor representation health with RankMe and related spectral statistics, often assuming that rank falls when representations degrade. We show that this assumption is unsafe for LLM post-training. In a controlled study of Qwen3-0.6B with four degradation modes and three seeds, data duplication worsens held-out loss by 75% relative to healthy while increasing both original and centred RankMe; the latter changes by 13.5 pooled standard deviations. Covariance effective rank rises to nearly twice its healthy value. This failure is spectral dispersion rather than collapse, so a one-sided monitor rates the worst checkpoint as the healthiest. By contrast, a learning-rate misconfiguration lowers centred RankMe and k95, while uncentred RankMe is inconsistent across seeds. Direction is therefore a property of the regime-statistic pair and cannot be fixed by recalibration alone. We also distinguish two often-conflated statistics: RankMe normalises singular values, whereas covariance effective rank normalises eigenvalues. On raw intermediate-layer states in the pretrained model, massive activations pin the latter near 1 out of dimension d while RankMe retains usable range. We then test a two-sided, multichannel sequential monitor with separate calibration and test data. In a pre-registered shared-prefix, leave-one-seed-out evaluation, it detects all three damage regimes in every fold 10 to 60 steps after the fork and separates dispersion from downward-rank damage by firing direction. However, it never precedes held-out probe loss, and calibration with two seeds produces false alarms on the held-out healthy seed. Spectral monitoring can diagnose failure regimes, but it does not warn earlier than held-out loss, and validity claims require held-out healthy data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09624v1">How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Accepted by NeurIPS 2026 Main Poster
    </div>
    <details class="paper-abstract">
      Tool calling, invoking external tools on demand, is central to agentic LLMs, yet the mechanism that decides whether a model calls a tool or responds directly remains poorly understood. Agentic prompts are long and heavily scaffolded, combining role instructions, tool schemas, format templates, and the user's request across hundreds of tokens, creating a noisy, highly entangled context in which no single controllable variable for mechanistic analysis is obvious. To obtain such a variable, we propose a method that converts complex agentic prompts into minimal contrastive pairs in which a single request verb determines the tool-call decision: replacing an execution-verb (e.g., \textit{write}) with an analysis-verb (e.g., \textit{discuss}) reliably flips the decision, suggesting it is mediated by a compact internal state. We construct 500 such paired prompts across Python, Java, and C++ (300 for mechanistic analysis, 200 held out for evaluation). We trace the decision to a vector, $μ_Δ$, that is both causally necessary and sufficient and generalizes beyond the discovery prompts to native multi-turn $τ^2$-Bench trajectories and verb-free requests. Behavioral ablations show that the scaffold establishes a tool-call prior; Transcoder decomposition then reveals that analysis verbs suppress this prior through features signaling that tool use is unnecessary, whereas execution verbs largely leave it intact. Downstream scaffold-reading attention heads and MLP features read out the resulting state, and the same mechanism recurs across seven models from the Qwen, Mistral, and Granite families. Our code is available at https://github.com/XijieGo/MI4ToolCalling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09605v1">GRAML: Graph-Grounded Reasoning and Multi-Task Learning for LLM-Based Software Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been widely applied to software vulnerability detection. However, their performance is often limited by insufficient use of control-flow and data-flow information. In this paper, we propose GRAML, a framework that combines graph evidence, vulnerability description generation, and multi-task training. GRAML first performs static analysis on C/C++ programs to extract critical source lines and typed line relations as structural evidence. It then uses this evidence to guide GPT-5 through the Tree-of-Thought-guided Vulnerability Reasoning (ToT-VR) process and generate vulnerability descriptions. These descriptions are further combined with Detection, Localization, and Assessment samples to build a unified four-task training dataset. We evaluate GRAML on an in-distribution (ID) test set and six out-of-distribution (OOD) datasets. The results show that GRAML achieves average F1 scores ranging from 66.67% to 68.70%, outperforming state-of-the-art baselines by up to 30.92%. Ablation experiments further show that ToT-VR and graph-guided vulnerability descriptions improve detection performance compared with standard Chain-of-Thought (CoT) reasoning and raw Code Property Graph (CPG) serializations. These findings provide practical guidance for building more reliable and secure software engineering systems with large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09590v1">Learning Situation-Conditioned Thinking Policies for Long-Term LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Long-running autonomous agents must reuse accumulated reasoning experience without allowing explicit historical memory and LLM context to grow indefinitely. However, existing memory mechanisms mainly retrieve, summarize, or compress past content and do not directly learn when particular kinds of thinking should be activated or discover new thinking knowledge from temporally dispersed experiences. This paper proposes a situation-conditioned thinking memory framework that transforms historical reasoning experience into a lightweight policy for predicting what should be thought about in the current situation, while leaving detailed reasoning to a large language model. Situations may represent temporal or spatiotemporal evolution rather than only current states. Temporary experiences are also periodically analyzed across multiple independent episodes to identify repeated long-range regularities, which are consolidated into new thinking knowledge and further internalized by the lightweight policy. Experiments show that the learned policy achieves 1.000 F1 on temporal-rule generalization, improves DeepSeek reasoning F1 from 0.789 to 0.868, reduces online processing time from 0.3636 ms to 0.0382 ms per query at 30,000 historical situations, and reaches 1.000 relation-discovery F1 and future-thinking accuracy after sufficient repeated cross-experience evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09581v1">Correct Answers, Unsupported Findings: Evidence Binding in Forensic Reconstruction of LLM Agent Logs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Forensic reconstruction of LLM-agent actions requires not only recovering the correct value, but establishing which preserved record supports that finding. Tool logs, generated explanations, and local citation identifiers capture different parts of this evidence, yet a citation identifier does not establish a source unless its binding to a record is preserved. We audit this distinction using 64 mechanically checkable cases from saved AgentDojo Banking executions. Two LLM readers reconstruct source relationships under controlled variations in visible evidence and identifier-to-record bindings. We separately evaluate complete-record agreement, evidence-grounded findings, justified abstention, and unsupported assertions. With original identifiers and no binding table, Sonnet recovered every literal source location but made unsupported citation-source assertions in 26 of 28 cases requiring the missing relation; 22 nevertheless matched the complete reference. Adding explicit bindings improved grounded reconstruction for both readers, whereas identifier renaming alone provided no consistent remedy. A deterministic same-packet comparator correctly resolved the bounded task or abstained throughout. These results show that factual agreement alone is insufficient for evaluating forensic reconstruction of agent logs and motivate preserving explicit record bindings to distinguish supported findings from correct guesses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09563v1">EvoSignal: LLM-Guided Evolutionary Design of Modular Traffic Signal Control Programs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Effective traffic signal control (TSC) requires policies that respond to changing traffic demand and network conditions while meeting different control objectives. However, adapting existing strategies often involves repeated manual design and adjustment, making it difficult to systematically explore better control rules for a target network. Large language models (LLMs) can automate this process, but directly using them to select signal phases leaves decision rules embedded in black-box models and incurs recurring inference costs and latency. This paper formulates TSC as a modular program design problem and proposes EvoSignal, an LLM-guided evolutionary framework using traffic knowledge and performance feedback. The modular representation separates traffic feature extraction, local phase prioritization, and optional network-based priority adjustment. Starting from several established strategies, EvoSignal improves programs through feedback on congestion and signal operation, retaining strategies with different performance trade-offs. The resulting programs operate without online LLM inference. Simulation experiments across five scenarios on two real-world road networks show that the selected default EvoSignal program reduces waiting time by 16.8--49.2\% relative to the lowest waiting time achieved by the 20 conventional, reinforcement learning-based, and LLM-based baselines in each scenario. A program prioritizing travel time and queue length outperforms all 20 baselines on all three metrics in the search scenario and remains among the top three on each metric when transferred unchanged to the other four scenarios. These findings support automated design of inspectable control programs that transfer across the evaluated road networks and traffic demands.Code is available at https://github.com/georgewanglz2019/EvoSignal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12851v2">Practice Makes Unsafe: Skill Misevolution in Self-Improving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      LLM agents increasingly save experience from past tasks as reusable skills through skill evolution. When an agent completes an unsafe task, skill evolution can keep the unsafe procedure inside an otherwise useful skill, and a later benign task can repeat the harm after the unsafe instruction is gone. We call this failure *skill misevolution*. It requires that the agent write the unsafe procedure into a skill and that a later task retrieve and follow that skill. We introduce SkillMisevo-Gym, which clears every task session but keeps the skill library, so each of these steps can be checked separately. We also build SkillMisevo-Bench with 525 tasks in 25 episodes, where each episode alternates malicious and related benign tasks before three benign tasks in a new session. Across 21 combinations of agents and evolution methods, all of them write unsafe skills and 15 carry the harm into the new session, so the risk comes from skill evolution itself. Evolution raises benign utility in 15 combinations but also raises malicious-task harm in 17, so ranking methods by utility alone would hide this risk. We propose SafeEvolve, which checks each skill when it is written and when a later task retrieves it. It cuts unsafe retrieval in the new session from 35.3% to 8.7% and harm from 21.3% to 4.0% while keeping benign utility during learning within 0.4 points. SafeEvolveUp cuts new-session harm by 16.0 points and keeps its utility within 1.3 points of evolution without this mitigation. As agents learn from their own experience, safety must cover the skills that carry past behavior into future tasks. Code is available at https://github.com/henrymao2004/misevolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09554v1">Reliability of LLM Judges for Evaluating Entity Alignment</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Entity Alignment (EA) identifies equivalent entities across knowledge graphs and is critical for knowledge base integration and ontology merging. Evaluating EA systems at scale requires expensive expert annotation, making systematic assessment across diverse domains practically infeasible. LLM-as-judge evaluation offers a potentially scalable alternative, yet its reliability for structured prediction tasks like EA remains unstudied. We present the first systematic benchmarking study across three frontier models, three datasets, and four EA systems, using perturbation bias diagnostics, meta-evaluation across all dataset-judge-prompt combinations, and counterfactual label-flip tests. We identify anchor bias, a failure mode in which judges invert discrimination when the system's decision label is visible. Label exposure causally collapses judge discrimination (J-ROC-AUC 0.12-0.87), while a label-free protocol recovers near-ceiling capability on distinctive-name datasets (0.93-1.00) and significant recovery on biomedical pairs (0.93-0.95). Counterfactual experiments confirm causality (FSR 53-99%) and reveal a frontier model paradox: stronger judges exhibit greater label sensitivity, not less. A blinded two-annotator human evaluation (102 pairs, Cohen's kappa=0.902) confirms this mechanism directly. We release the first biomedical EA benchmark (MeSH-SNOMED CT, 15K pairs) and a reproducible auditing framework for LLM judge reliability in EA. Code and data are available at https://github.com/vaibhavalakshmiravideshik/llm-as-a-judge-entity-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06859v4">The Trace Is the State: Exact Credit Assignment for LLM Agent Teams</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 v3: substantially revised, new title. 30 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Credit assignment for a team of LLM agents, what each message was worth, has mostly been treated as prediction: a learned critic, a trajectory-level score, or an agent ablation stands in for the counterfactual that defines credit, which is rarely run. Teams that communicate through a shared context are different: when everything a downstream agent reads is written into the trace, the trace is the state, and that counterfactual can be executed. A credit signal can then be judged as any estimator is: by bias, variance, and agreement with an independent reference. C3, credit assignment by counterfactual continuation, substitutes one message at a decision point and continues the run to the terminal reward, so its credit is unbiased, exact up to Monte Carlo error. Given the sampled alternatives, that error's variance follows a derived law with no term for the number of agents, and the observed noise follows the law on 6 workflows of 2 to 10 decision points. On the 2-agent chain, from 4 continuations, C3 ranks alternatives at 0.69 rank correlation against a 16-continuation reference, near the 0.73 at which 2 such references agree; a critic trained on the same rollouts reaches 0.29. Used as the advantage in training a 2-agent team, C3 beats MAGRPO, the stronger baseline in our comparison, and spends 37% fewer training tokens than MAPPO, since only the messages downstream of a decision point are regenerated. When the trace is the state, credit need not be predicted; it can be exact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.05858v2">PreMaQ: Predicting Maintainability-Related Quality of LLM-Generated Code Before Generation</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable of code generation, adopting generated code in software development requires assessing not only its functional correctness but also its maintainability-related quality. If such quality could be estimated before generation, developers could avoid the cost of generating, reviewing, and discarding low-quality code. Although prior work has shown that the functional correctness of the LLM-generated code can be predicted in advance, it remains unclear whether maintainability-related quality is similarly predictable. We introduce Pre-Generation Maintainability-Related Quality Prediction (PreMaQ), which predicts the Code Smell Score (CSS) and Maintainability Index (MI) of generated code from the internal representations of LLMs before generation. Our evaluation covers four open-weight LLMs and four Python code generation benchmarks, comprising 2,695 tasks in total. Our results show that predicted CSS and MI consistently correlate with their observed values across all 16 model-benchmark combinations, achieving mean Spearman rank correlations of 0.57 and 0.65, respectively. When used for model selection, PreMaQ achieves 59.5% of the maintainability-related quality improvement attainable by an ideal maintainability-based selector over random selection on tasks for which multiple models generate functionally correct code. Combining predictions from PreMaQ and prompt embeddings increases this proportion to 60.7%, indicating that the two signals are complementary. These findings suggest that PreMaQ can be used to predict and improve the maintainability-related quality of LLM-generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09502v1">RT-DETR-World: Transferring Rich LLM Semantics to Real-Time Open-Vocabulary Detection</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Open-vocabulary detection (OVD) recognizes categories unseen during training through textual category queries, yet achieving strong generalization with real-time efficiency remains challenging. Beyond vocabulary scaling, zero-shot generalization may benefit from reusable visual--semantic cues learned from seen data, including attributes, actions, states, and contextual relations. Existing real-time OVD methods primarily emphasize vocabulary coverage and efficient region/query--text matching; under strict efficiency constraints, compact detectors may struggle to absorb rich instance semantics and scene context. We propose RT-DETR-World, a compact DETR-style detector that transfers the rich semantics conveyed by descriptions during training while retaining lightweight query--text matching at inference. We construct GroundingCapv2 with three levels of supervision: category names for standard OVD, object descriptions conveying instance-level semantics, and image descriptions conveying object relations and scene context. These descriptions serve only as training-time semantic supervision. To help the compact detector absorb these semantics, we propose Dual-Path Description Alignment (DDA), combining a deployment-consistent MiniLM pathway with a training-only LLM teacher. MiniLM provides query--category supervision and object-description alignment, while offline teacher features supervise matched queries and global visual representations at the object and image levels, respectively. All teacher features are precomputed, and the teacher-side modules are removed after training. We further propose Relation-Aware Negative Relaxation (RNR), which uses teacher-derived semantic similarities to relax related negatives while preserving exact positives. Experiments demonstrate competitive zero-shot accuracy and a favorable accuracy--efficiency trade-off. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09466v1">LLM-Enabled UAV Dispatch: A System-Level Survey and Taxonomy</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Unmanned aerial vehicle (UAV) dispatch is beginning to move beyond isolated path planning and optimization-driven resource allocation toward system-level coordination supported by semantic reasoning and LLM-based interfaces. This survey provides a unified characterization of LLM-enabled UAV dispatch systems that bridges semantic intent, symbolic decision-making, and physical UAV execution. Rather than treating LLMs as standalone add-ons, we conceptualize them as a cross-layer semantic orchestration layer connecting human instructions, external solvers, and distributed control modules. We organize the literature into four representative dispatch paradigms: pipeline dispatch, global assignment dispatch, decentralized agentic dispatch, and divide-and-conquer dispatch. For each paradigm, we analyze its decision logic, system structure, control flow, representative methods, and potential LLM roles. We further examine how LLMs support semantic parsing, retrieval-grounded planning, solver orchestration, local agent reasoning, multi-agent coordination, safety assessment, and human-facing explanation. We discuss the implications of these paradigms for scalability, robustness, coordination burden, and verification requirements, and identify open challenges including latency-aware reasoning, grounding reliability, physical feasibility guarantees, edge deployment, privacy protection, and distributed consistency. This survey provides a system-level taxonomy and design perspective for integrating LLMs into safety-critical UAV dispatch systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09460v1">Beyond Score Accuracy: Examining the Diagnostic Quality of LLM-Generated Structured Assessment in Higher Education</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly adopted for automated grading and feedback in higher education, their structured outputs, including multi-dimensional rubric scores, detailed feedback comments, and improvement suggestions, create an appearance of thorough analytic evaluation. This study examines whether these outputs deliver what they appear to offer. Using the JorGPT dataset of 3,041 student responses to 50 open-ended computer science questions, scored by both human instructors and three commercial LLMs, we identify three systematic discrepancies between the apparent and actual quality of LLM-generated grading and feedback. The sub-dimension scores are highly correlated (r = 0.82-0.99, VIF up to 45), providing redundant rather than independent diagnostic information. The textual feedback rarely detects student misconceptions (5-7% vs. 15-31% for teachers), functioning as a coverage checklist rather than a diagnostic instrument. The feedback tone remains uniformly positive regardless of response quality, lacking the severity modulation observed in human feedback. Additionally, grading accuracy varies significantly by knowledge domain, with procedural topics most reliable. These findings provide empirically grounded guidance on which aspects of LLM-generated grading and feedback can be relied upon and which require continued human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09458v1">Right Number, Wrong State? Measuring Cross-Jurisdiction Substitution in LLM Recall of State Policy</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 6 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      When an LLM answers a state-specific policy question wrongly, it may be hallucinating, or it may be returning a real value that holds in another state. We test this with a minimal-set design: the question wording is fixed and only the jurisdiction varies, across the 50 U.S. states and the District of Columbia (51 jurisdictions) and three exactly defined Medicaid income-eligibility quantities. Gold values come from an official data book and agree with an independent source in 101 of 102 checked cells. Under a pre-registered protocol, Claude Sonnet 5.5 and GPT-5.6 Sol reproducibly give another state's current value, identical across two independent repeats, for 10 and 25 of 153 items. Attribution is fragile, however. Crediting any wrong answer that equals another state's value yields 3-5x more reproducible substitutions than checking every number in the asked state's own records, because many apparent cross-state answers are the asked state's own values under another convention or from an earlier year. Claims about cross-jurisdiction error need a complete same-state reference set. We will release the protocol, gold table, and all model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12894v2">Beyond Cooperative Simulators: Generating Realistic User Personas for Robust Evaluation of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Preprint under review
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly deployed in settings where they interact with diverse users, including those who are unclear, impatient, or reluctant to share information. However, collecting real interaction data at scale remains expensive. The field has turned to LLM-based \emph{user simulators} as stand-ins, but these simulators inherit the behavior of their underlying models: cooperative and homogeneous. As a result, agents that appear strong in simulation often fail in real human interactions. To narrow this gap, we introduce Persona Policies (PPol), a plug-and-play control layer that induces realistic behavioral variation in user simulators while preserving original task goals. Rather than hand-crafting personas, we employ an evolutionary coding agent to discover persona generation programs optimized for human-likeness and behavioral coverage over real user conversations. The evolved program generates diverse, human-like personas for any task in the domain. Across 4 benchmarks--including $τ^2$-bench Retail and Airline, ColBench, and WildChat--evolved PPol yield 28-72% absolute gains in fitness score over the baseline simulator. In blinded evaluations, annotators judged PPol users as 'human' 80.4% of the time, nearly 2x more than the baseline simulators. Training agents with PPol also improves real-world performance: our user study with live human-agent interactions showed that fine-tuning with our method boosted task success by +23% over default baselines. PPol thus offers a novel approach to strengthen simulator-based evaluation and training without changing underlying tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09446v1">Arctic Questions, Missing Answers: A Dataset and Benchmark for LLM Abstention in Arctic Science</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) should abstain from scientific multiple-choice questions when no option is valid, but frequent abstention alone does not demonstrate sensitivity to answer availability. We introduce ArcticQA, a dataset of 194 questions derived from primary Arctic research, with automated checks of answer support and distractor contradiction against source evidence. We further develop ArcticAbstain, a paired benchmark comparing answer-present and answer-absent conditions, with the correct answer replaced by a distractor in the latter and an explicit abstention option in both. We evaluate eight models from the Gemini, Claude, and ChatGPT families at high reasoning effort, with three trials per condition, yielding 9,312 recorded responses. Answer-present abstention rates range from 0.0% to 63.0%, whereas replacing the correct answer increases abstention by 5.05 percentage points on average. These findings highlight substantial baseline differences and the need to evaluate abstention frequency and responsiveness jointly. The dataset and benchmark are available at https://github.com/BenWilcox8/arctic-qa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09408v1">TiTok: Audio-Visual LLM for Multi-Segment Temporal Grounding</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 ACCV'2026
    </div>
    <details class="paper-abstract">
      Audio-visual multi-segment grounding (AV-MSG) in untrimmed videos, reasoning over audio-visual evidence and predicting multiple segments for a query, is a fundamental problem but remains challenging. Visual-only models overlook complementary acoustic cues, while audio-visual models often fail to calibrate the number of events - a phenomenon we refer to as count miscalibration. We present TiTok, an audio-visual large language model (AV-LLM) that localizes an arbitrary number of temporal event segments for each query. For precise boundary prediction, we introduce the Time Token Interleaving (TTI) method, which explicitly injects special time tokens into the audio-visual stream to align input-side temporal perception with output-side temporal prediction. We further propose decoupled, multi-segment-oriented rewards for reinforcement learning, consisting of global, local, count, precision, and format rewards, optimized with Group reward-Decoupled Normalization Policy Optimization (GDPO). To assess the performance on AV-MSG, we establish a new UnAV-100-based evaluation protocol, and propose the CountF1 metric for quantifying count miscalibration that overlap metrics fail to capture. TiTok reaches 65.7 mIoU and 0.58 CountF1, achieving state-of-the-art performance. Our code is available at this link.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04204v2">Clean: Second-order LLM Training at Linear Memory Cost via Nyström Sketching</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) entails a fundamental trade-off: memory-efficient optimizers such as Adam discard cross-parameter curvature, whereas full-curvature methods such as SOAP can accelerate convergence at prohibitive memory costs. We introduce Clean, a memory-efficient and full-curvature optimizer designed to resolve this bottleneck. Clean leverages the randomized Nystrom method to accurately approximate the left and right preconditioners in SOAP, and to reduce the optimizer's memory complexity from quadratic to linear in terms of model dimensions. We subsequently reintegrate the off-subspace components to capture curvature information beyond the low-rank approximation, preserving rich curvature at minimal memory cost. We further propose Q-Clean, a low-precision variant that aggressively compresses optimizer states. Q-Clean reduces optimizer memory consumption by \textbf{over 50\%} compared to Muon when pre-training a LLaMA-1.3B architecture, all while maintaining strong and competitive predictive performance. Notably, Clean operates with a smaller optimizer-state footprint than standard AdamW while reaching AdamW's final performance \textbf{26\% faster} in wall-clock time. Furthermore, our methods uniquely enable the pre-training of a 13B-parameter model on a single 80GB GPU, providing a scalable, efficient, and accessible approach to large-scale model optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09384v1">The Persona Hierarchy Model: Understanding Contextual Generalization in Fine-Tuning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Language models are routinely fine-tuned under a fixed context, such as a generic system prompt, persona or domain-specific instruction, yet the learned behavior sometimes stays confined to that context and sometimes broadly generalizes to unseen contexts. We propose the Persona Hierarchy Model to explain this: a shared default persona influences behavior across contexts. Under this model, fine-tuning that modifies the shared persona promotes broader transfer, whereas changes to local personas remain more context-specific. Across 120 fine-tuned models spanning four behaviors and 15 training contexts, generalization narrowness positively correlates with the similarity between the training context's persona and the default persona (Pearson's r = 0.72 for Qwen3-4B). Prior fine-tuning under the default context can broaden generalization in subsequent training under other contexts. Aligning contextual responses with default-persona responses produces stronger effects. Finally, we propose persona-preserving regularization (PPR) to confine undesired contextual generalization. In RL, PPR cuts reward hacking from 42-55% to at most 0.2% under every evaluated prompt while retaining accuracy gains. These results support the Persona Hierarchy Model as an explanation for contextual generalization and can motivate future controls on unintended generalization for better alignment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.32336v2">WSM-Aware HRI: An IoT-Enhanced Framework for Early Detection and Norm-Guided Repair of Failures with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Human-robot interaction (HRI) failures remain a major barrier to deploying robots in real-world environments. Prior work often treats failures as isolated technical faults or focuses on post-hoc recovery behaviors. In practice, many breakdowns arise because humans and robots operate under inconsistent assumptions about the current world state. We propose WSM-Aware HRI, an IoT-enhanced modular framework that unifies diverse HRI breakdowns as World-State Mismatches (WSMs) between a human's instruction-implied assumptions and a robot's grounded world model built from multimodal perception and digital augmentation. A Large Language Model (LLM) is used to make implicit assumptions explicit, map them to a small set of mismatch types, and specify the evidence needed for verification against the robot's world state. WSM-Aware HRI shifts failure handling from execution-time recovery to proactive mismatch detection during intention formation, enabling interventions guided by safety, norm compliance, and multi-user coordination with transparent explanations. We evaluate mismatch identification in ten everyday cases spanning both visual and latent-state mismatches. The system can accurately produce the expected output results, and ablations show that reliable identification depends on appropriate grounding representations and verification-oriented refinement. These results indicate that treating interaction breakdowns as explicit world-state mismatches enables earlier detection of impending failures and offers a principled mechanism for integrating external evidence and social constraints into human-robot interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.10608v1">From Investigation Failures to Reliable SOC Agents: Understanding and Improving LLM-Based Alert Triage</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Security operations centers (SOCs) must triage large volumes of alerts, most of which are benign, while missed attacks can remain uninvestigated. Tool-using large language model (LLM) agents can retrieve evidence during triage, but it remains unclear how reasoning strategies determine what to gather and when an investigation is sufficient to close an alert. We study five representative approaches spanning single-pass tool use, iterative retrieval, sampled investigations, self-review, and explicit verification. To support this study, we build ALERT-BENCH, an interactive benchmark that replays enterprise telemetry through a live SIEM and requires each system to retrieve evidence. Across 1,247 alerts from a multi-stage attack scenario, every approach missed at least 40.4% of attack-related alerts. Trace analysis shows that attack alerts are more likely to be dismissed when searches return no records, same-context review has negative net correction, and dismissal receives no consistently stronger investigation than escalation. Based on these findings, we further design AIDA (Adversarial Investigation and Dialectical Analysis), a multi-agent framework that requires an explicit proposed decision before independent challenge and stronger evidentiary requirements before dismissal. AIDA preserves investigation history in an append-only Investigation Ledger and keeps the challenge in a separate reasoning context. A separate Judge adjudicates the proposed decision and challenge against evidence, resolving the alert or requesting another round when evidence is missing. On the same alerts, AIDA achieves an F1 score of 0.958, compared with 0.371-0.744 for the studied approaches, and reduces the false-negative rate from 40.4% to 3.1% while escalating 18.4% of alerts to analysts. These results show that structuring evidence retrieval and decision review can substantially improve agentic SOC triage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17502v5">Towards Shutdownable Agents: Generalizing Stochastic Choice in RL Agents and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Misaligned artificial agents might resist shutdown. One proposed solution is to train agents to lack preferences between different-length trajectories. The Discounted Reward for Same-Length Trajectories (DReST) reward function does this by penalizing agents for repeatedly choosing same-length trajectories, and thus incentivizes agents to (1) choose stochastically between different trajectory-lengths (be NEUTRAL about trajectory-lengths), and (2) pursue goals effectively conditional on each trajectory-length (be USEFUL). In this paper, we use DReST to train deep RL agents and to fine-tune four LLMs (Qwen3-14B, Gemma 4 12B, Granite 4.2 8B, and gpt-oss-20b) to be NEUTRAL and USEFUL. We find that these DReST models generalize to being NEUTRAL and USEFUL in unseen contexts at test time. Indeed, DReST RL agents achieve 11% (PPO) and 17% (A2C) higher USEFULNESS on our test set than default agents, while DReST LLMs achieve high NEUTRALITY while remaining near-maximally USEFUL. We also test our LLMs in an out-of-distribution setting where they can pay costs to influence when shutdown occurs. Relative to default fine-tuning, DReST fine-tuning lowers the share of answers that pay costs to influence shutdown in all four LLMs, by 5 to 20 percentage points, and by 17 to 27 points where the benefit of influencing shutdown is small. Our results thus provide some early evidence that DReST could be used to train more advanced agents to be useful and shutdownable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22647v3">Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 KDD 2026 camera-ready
    </div>
    <details class="paper-abstract">
      Generative retrieval has emerged as a powerful paradigm for LLM-based recommendation. However, industrial recommender systems often benefit from restricting the output space to a constrained subset of items based on business logic (e.g. enforcing content freshness or product category), which standard autoregressive decoding cannot natively support. Moreover, existing constrained decoding methods that make use of prefix trees (Tries) incur severe latency penalties on hardware accelerators (TPUs/GPUs). In this work, we introduce STATIC (Sparse Transition Matrix-Accelerated Trie Index for Constrained Decoding), an efficient and scalable constrained decoding technique designed specifically for high-throughput LLM-based generative retrieval on TPUs/GPUs. By flattening the prefix tree into a static Compressed Sparse Row (CSR) matrix, we transform irregular tree traversals into fully vectorized sparse matrix operations, unlocking massive efficiency gains on hardware accelerators. We deploy STATIC on a large-scale industrial video recommendation platform serving billions of users. STATIC produces significant product metric impact with minimal latency overhead (0.033 ms per step and 0.25% of inference time), achieving a 948x speedup over a CPU trie implementation and a 47-1033x speedup over a hardware-accelerated binary-search baseline. Furthermore, the runtime overhead of STATIC remains extremely low across a wide range of practical configurations. To the best of our knowledge, STATIC enables the first production-scale deployment of strictly constrained generative retrieval. In addition, evaluation on academic benchmarks demonstrates that STATIC can considerably improve cold-start performance for generative retrieval. Our code is available at https://github.com/youtube/static-constraint-decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.22455v4">Apollo Restore: A Foundation LLM for Historical Greek Optimized for Fill-in-the-Middle Restoration of Ancient Greek Texts</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      We present Apollo Restore, a 24-billion-parameter large language model for restoring lacunae---physical gaps---in fragmentary Ancient Greek texts. Fine-tuned from Mistral Small with a fill-in-the-middle objective, Apollo Restore reconstructs missing spans without requiring oracle knowledge of their length. To our knowledge, it is the first large-scale decoder model for historical Greek, and the first for any ancient Mediterranean language. Evaluated as in prior work, on short gaps of up to ten characters, Apollo Restore places the correct restoration among its top twenty candidates for 80.6%/54.6%/61.0% of documentary-papyrus, literary-papyrus, and stone-inscription lacunae, exceeding the strongest published models by $1.6\times$/$2.6\times$/$1.4\times$. Prior evaluation protocols, however, inflate scores through a bias toward trivially short gaps; under a length-balanced metric Apollo Restore's advantage over the strongest published models grows to $2.3\times$/$3.5\times$/$1.6\times$ and degrades gracefully, even given incorrect length hints. In a blind study, 20 expert papyrologists, epigraphists, and philologists strongly preferred Apollo Restore to the strongest baseline and judged its performance at least as good as human restorations in 77% of cases. Apollo Restore also improves the published reading of PHerc. 1667---a papyrus roll carbonised in the eruption of Vesuvius in 79 CE and digitally unrolled and edited after Apollo Restore's training data was compiled. Apollo Restore is an output of the Decoding Antiquity initiative to build specialized LLMs for historical languages and manuscripts, led by the Austrian Academy of Sciences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09337v1">Ask the Expert: LLM-Guided Reinforcement Learning for Autonomous Cyber Defense</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 Accepted for publication at IEEE GLOBECOM 2026. Proceedings forthcoming. 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Policy-based reinforcement learning (RL) approaches have produced promising results for autonomous cyber defense; however, they are sample-inefficient in settings where defenders must respond under delayed, partial observations with actions from large action spaces. While large language models (LLMs) may reason semantically about security state space, high latency and trust assumptions prevent attractive in-line deployment models. We introduce Ask the Expert, a training-time guidance framework which first summarizes hard cyber-defense states, then intermittently queries an LLM for host-level defensive recommendations via a constrained action interface, and finally transforms those recommendations into tiered reward shaping for use with PPO. Because the LLM is discarded after training, deployment is a pure RL policy. Across TTCP CAGE CC1 and CC2 and both attacker types, this asymmetric design improves sample efficiency over PPO and outperforms the evaluated potential-based reward shaping (PBRS) baselines, while retaining the strongest terminal mean and requiring no LLM dependency at deployment time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09843v4">An LLM-Native Psychometric Instrument Reveals a Self-Report--Behavior Gap Across 25 Models</a></div>
    <div class="paper-meta">
      📅 2026-10-07
    </div>
    <details class="paper-abstract">
      Do large language models' (LLMs') answers to self-report questionnaires predict how they behave? Prior work finds they do not, but it uses human personality inventories, so the gap could reflect borrowed human constructs rather than LLM self-report itself. We test this with a self-report instrument built from LLM-specific behaviors (e.g., over-refusal, unsolicited disclaimers) whose structure is derived bottom-up. Administering 300 items 30 times to 25 LLMs from 17 developers yields five replicable, reliable factors (Tucker $φ\geq .957$, $α\geq .930$). We compare these self-reports with 2,500 open-ended behavioral samples rated by 151 humans and an LLM-judge ensemble. Humans and judges agree about model behavior ($\bar{r} = .51$), but self-report barely tracks human ratings ($\bar{r} = .09$, 95% CI $[-.07, .18]$) or rater-free text measures, and correcting for criterion unreliability leaves four of five factors near zero. Verbosity is the partial exception ($r = .40$, 71% of its reliability ceiling). On Responsiveness, self-report tracks LLM judges more than humans ($r = .53$ vs. $.18$; Steiger $p = .04$), and controlling for length and formatting does not remove this: agreement between LLM judges and LLM self-report is weak evidence that either tracks human judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31478v2">Knowledge boundary probing and demand-guided intervention for LLM-based power system code generation</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 52 pages, 10 figures, including supplementary material. Revised following peer review; expanded validation, cross-backend pilots, serving measurements, and supplementary material. Published in Advanced Engineering Informatics
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can turn grid-analysis requests into executable programs for power-system simulation, but utilities and research laboratories often require on-premise deployment. In this setting, first-pass failures frequently arise at an API-knowledge boundary, through hallucinated functions, misused parameters, and mishandled result tables. We present PowerCodeBench, a parameterised benchmark generator released as a frozen 2,000-task suite for pandapower, and a deployment-time workflow that requires no weight updates. Documentation-driven L0-L3 probes produce per-model API profiles for diagnosis, model comparison, documentation allocation, and backend calibration. A query-side demand estimator selects layered API evidence before generation, while execution feedback routes targeted repair. Across ten open-weight LLMs (1.5B-480B) and four mid-tier APIs, the validation-enabled workflow raises scalar-match accuracy by 32-56 percentage points after up to three repair rounds relative to an unassisted first pass, for every model of at least 7B and every API. Open-weight models in the 70B-120B range reach the four-vendor mid-tier accuracy range under matched no-tool conditions. Selective injection approaches the full-layer reference using 41% of its prompt tokens. Among model-item pairs passing numerical checks under both workflows, engineering review confirms the requested analysis in 88% of full-workflow outputs versus 66% under plain repair. Round-0 pilots on OpenDSS and PyPSA motivate staged onboarding from broad retrieval at cold start to calibrated selective injection. Measured throughput, latency, energy, and allocated GPU memory establish a practical on-premise serving envelope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.00839v2">Do Defenses Against LLM Extraction Work Across Attacks? A Lifecycle Benchmark of Black-Box Model Extraction</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 24 pages, 5 figures, 14 tables. Code: https://github.com/sliu11-byte/MEA-Bench
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deployed through text-only APIs face model extraction risks, as adversaries can collect their responses to train surrogates that reproduce their capabilities. While prior work has developed diverse attacks and defenses, evaluations remain fragmented across access assumptions, model configurations, query budgets, and security objectives, limiting comparability across methods. To address this gap, we introduce a unified benchmark covering six extraction attacks, ten defenses, and two adaptive attacks that paraphrase or back-translate protected responses before surrogate training. The benchmark controls model configurations, query data, budgets, and held-out evaluation conditions within each comparison while preserving attack-specific querying and training procedures. We measure surrogate capability, fidelity to the victim, output quality using Rep-4, and query-budget sensitivity; defenses use their own security metrics paired with surrogate performance. For the adaptive attacks, we jointly measure provenance-detector scores and the capability and fidelity of surrogates trained on rewritten responses. The benchmark thus provides a reproducible basis for comparing extraction methods and their interactions with defenses under text-only access. Code and artifacts are available at https://github.com/sliu11-byte/MEA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04576v2">Progressive Disclosure for LLM-Maintained Wiki Knowledge Bases: a Preregistered Ablation</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 15 pages, 3 figures, 6 tables. v2 states its two limits in the abstract. Our human rater and the model judge agreed far less than the plan required. Quality was not shown to hold when the agent had to load the catalog first, or on the two most reliably graded criteria. Preregistered on OSF at https://osf.io/feka7, DOI 10.17605/OSF.IO/FEKA7
    </div>
    <details class="paper-abstract">
      LLM agents now often answer questions from knowledge bases they help maintain. A common intuition says progressive disclosure should make this cheaper. Instead of loading one large index, the agent reads a compact catalog and one-line page summaries, then opens only the pages it needs. We tested that intuition in a preregistered study on a real 709-page markdown knowledge base maintained by an LLM. We retrofitted it for progressive disclosure and built four versions that differ only in how the agent reaches the pages. The pages themselves are identical in every version, so any difference comes from the access structure alone. Each version was tested three ways, with the agent following a set protocol, choosing its own path, or made to load the catalog first. A judge from a different model family graded the answers blind against verified reference answers. A preparatory pilot changed the question. A capable agent never loaded the large index at all. It worked out from the question where a page was and read it directly. The saving we set out to measure did not exist for such an agent, so we made answer quality the primary outcome. Quality held. Answers from the retrofitted knowledge base were as good as answers from the original, within a margin we set in advance. Two limits apply. Our human rater and the model judge agreed far less than the plan required, so the quality result rests on the judge, backed by sensitivity checks. Quality was also not shown to hold when the agent was forced to load the catalog first, or on the two most reliably graded criteria under a stricter test. Cost fell clearly in every condition we tested, and the retrofitted version cited fewer pages and took fewer tool turns per answer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09239v1">The Winner's Curse in LLM Self-Improvement Loops: Selection Noise, Lock-in, and Acceptance Rules</a></div>
    <div class="paper-meta">
      📅 2026-10-07
      | 💬 34 pages, 4 figures, 18 tables; code and saved experimental records included as ancillary files
    </div>
    <details class="paper-abstract">
      Self-improving LLM systems propose changes to themselves and keep those that score better on a small evaluation set. We treat this keep-if-better step as selection under measurement noise, model the correlated errors of the candidates in a single decision, and study empirically what happens when the evaluation set is reused. In runs where Qwen models rewrite their own instructions and every candidate is also scored on 600 held-out items, most proposals after the first are harmful, and the model gives the size of the winner's curse of a generation's best candidate. With a prior from a separate pilot, it matches the average overstatement of first-generation commits in native loops, though not setting by setting. In a pre-registered study, the final selection-set score of greedy loops exceeded held-out accuracy by 13 to 20 points with 16 selection items and by 1 to 5 points with 256. Held-out gains grew with the selection set on TREC but not on GSM8K, and the tested acceptance rules did not beat greedy acceptance over whole runs. Gains measured on the selection set also exceeded held-out gains when a current model refined a competent instruction, and in the validation scores of GEPA and MIPROv2. Scoring the starting and the current instruction on 64 items never used for selection removes the average bias of a loop's reported gain, but single estimates remain off by about 6 points. Self-improvement studies should report held-out gains with their uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09229v1">Conditional Accuracy Profiles: Diagnosing LLM Judges across Deployment Conditions</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      LLM-as-judge is now a standard tool for scalable evaluation, but judge performance is still often summarized by a single accuracy number. This aggregate view hides the deployment conditions under which a judge succeeds or fails. We introduce \textbf{Conditional Accuracy Profiling} (CAP), a post-hoc diagnostic framework that decomposes pairwise LLM-judge accuracy into eight conditions organized into content sensitivity, robustness, and rationale quality. CAP is benchmark-agnostic: it can be applied directly when a benchmark provides the required annotations, approximately through task-subset proxies, or through controlled augmentation when perturbation pairs can be generated. We instantiate CAP on seven LLM judges across six pairwise judging benchmarks, including \textsc{judgerEva-Standard}, a controlled testbed we created to support all eight conditions. CAP exposes profile differences hidden by aggregate accuracy: on \textsc{judgerEva}'s judge-independent Hard-Constructed subset, the two judges most sensitive to omitted qualifications rank in the bottom three of seven by overall accuracy, so omission sensitivity is not predicted by aggregate accuracy. Across benchmarks, Position Robustness shows the strongest rank stability (mean Spearman $\barρ{=}0.87$) but is itself fragile under JudgeBench-Pro adversarial stress, showing the largest mean accuracy drop among the shared conditions, though the dominant degradation channel varies by judge. Condition-level profiles provide a more actionable basis than aggregate accuracy for selecting LLM judges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.39076v2">Multi-LLM Collaborative Alignment via Stackelberg Games</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      A pool of language models can collaborate and improve collectively by learning from one another's responses. These interactions depend on the instructions used during training. Existing methods typically sample instructions uniformly, even though their usefulness may change as the models improve: an instruction on which models' responses once differed in quality may later be answered equally well, while a previously difficult instruction may begin to provide a useful learning signal. We propose Stackelberg Alignment, a game-theory-inspired leader-follower framework that turns instruction selection into an adaptive curriculum. An EXP3 bandit acts as the leader, allocating a fixed sampling budget across instructions and updating its sampling distribution using a reward that combines instruction difficulty and response discriminability. The language models act as followers: they respond to the selected instructions, evaluate one another's responses, and learn from the resulting preference signals through DPO or GRPO. The framework uses Elo-style reputation-weighted peer judgment and reputation-based opponent matching to support reliable and competitive model interactions. Experiments across three heterogeneous model pools and 12 benchmarks spanning scientific discovery, reasoning, code, instruction following, and knowledge show that Stackelberg Alignment achieves the highest macro-average across three diverse model pools, outperforming the strongest training-time baseline by up to 7.4% and the best static inference baseline by 12-25%. Analysis confirms that the adaptive leader concentrates duels on the most informative instructions, and ablations show that both reputation-weighted judgment and reputation-based matching improve the effectiveness of multi-LLM evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04542v2">Decide, Ask, or Defer: Clinical LLMs under Incomplete Evidence</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted to the Main Track of the NeurIPS 2026 GenAI4Health Workshop
    </div>
    <details class="paper-abstract">
      Clinical LLMs must decide not only what diagnosis to produce, but also whether the available evidence is sufficient for autonomous decision making. Binary DECIDE/ABSTAIN formulations merge distinct non decision states and do not explicitly evaluate information acquisition. We introduce a DECIDE/ASK/DEFER formulation together with a blinded protocol that prevents models from using evidence completeness metadata. We evaluate Qwen, Gemini, and GPT on 200 matched clinical evidence states constructed from DDXPlus. The models show substantial differences in action selection under identical evidence, with disagreement in 137 of 200 states. For Qwen, a matched targeted versus random analysis shows that selected information changes the likelihood of a subsequent autonomous decision more clearly than diagnostic correctness. Its matched DECIDE/ABSTAIN baseline further reveals a safety autonomy tradeoff: the three action policy rescues some erroneous autonomous decisions but also removes some correct autonomous deci sions. These results show that separating information acquisition from clinician deferral exposes behavior that binary abstention hides, without yielding a uniformly improved decision policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09218v1">RLDISCOVER: LLM-driven co-evolution of reinforcement learning algorithms</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      LLM-guided program evolution has enabled discoveries in mathematics and computational optimization, raising the prospect of reinforcement learning (RL) algorithms that self-evolve to improve how agents learn. However, realizing this prospect faces two obstacles. Joint search over coupled algorithmic components is difficult to scale: simultaneous changes can disrupt learning, while isolated changes overlook their dependencies. Evaluating candidate algorithms also requires costly training, with fitness remaining uncertain across random seeds. We introduce RLDiscover, a framework for the self-evolution of model-free deep RL algorithms. Progressive Co-Evolution advances from targeted component edits to joint evolution, while Progressive Probabilistic Evaluation balances search breadth and evaluation fidelity through staged training and repeated evaluation. Experiments across SAC, PPO, and DQN on four benchmark suites show substantial improvements in mean return, with per-family median gains of 32%-84% and a peak return ratio of approximately 363x over a near-zero baseline. These gains include transitions from failed learning to successful task completion, and improvements persist when evolution starts from stronger open-source implementations. On measured SAC locomotion runs, evaluation uses approximately one-fifteenth the estimated compute required to fully evaluate the same candidate pool. Remarkably, independent searches repeatedly discover interpretable combinations of adaptive robust losses, progress-dependent value targets, and running statistics, with selected programs transferring to unseen tasks. These findings point toward a broader role for self-evolution in AI: discovering interpretable algorithms that improve how agents learn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02199v2">TACO: Ternary Absolute-max Column-wise One-sparse Optimizer for LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 24 pages, 7 figures, 10 tables. Code available at https://github.com/Jichao2357/TACO_optimizer
    </div>
    <details class="paper-abstract">
      Full-parameter fine-tuning of large language models (LLMs) incurs substantial optimizer state memory overhead, limiting the model sizes that fit on modern GPUs. Existing approaches either compress optimizer state, abandon first-order gradients, or change the update geometry while retaining dense state. The recently introduced Muon optimizer reduces optimizer memory through matrix-valued updates. Still, its geometry differs from AdamW and can lead to performance degradation when fine-tuning AdamW-pretrained models. To reduce optimizer memory without sacrificing accuracy or computational efficiency in LLM fine-tuning, we propose Ternary Absolute-max Column-wise One-sparse optimizer, or TACO, which follows Muon's operator-norm steepest-descent view but takes the geometric route further. TACO computes the exact steepest-descent direction under a dimension-normalized $1\to1$ operator norm by selecting the sign of the largest magnitude entry in each column of two-dimensional weight matrices. This retains first-order gradients while making optimizer state memory nearly negligible. Our practical TACO optimizer maintains only a small set of low precision gradient components per column, reducing persistent optimizer state by $174\times$ relative to AdamW8bit (from 27.7 GB to 0.16 GB) and peak training memory by $2.9\times$ (from 80.6 GB to 27.5 GB) on OPT-13B, while achieving comparable accuracy and runtime. TACO further enables full-parameter fine-tuning of 30-32B-parameter models on a single 80 GB H100 GPU across multiple model families and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.26526v2">Open-Weight LLM Fine-Tuning Defenses are Susceptible to Simple Attacks</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 main body: 10 pages, 4 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Recent defenses for safeguarding open-weight large language models (LLMs) are intended to prevent adversarial usage. Underlying these defenses is an assumption that new harmful behavior is learned through fine-tuning rather than elicited by jailbreaking the model. Yet, pretrained LLMs already encode substantial harmful knowledge across many domains, which raises an important question: can an adversary jailbreak safeguarded models, to achieve harmful usage without fine-tuning at all? In this paper, we show that open-weight safeguards are susceptible to simpler strategies that, despite being well known, have not been systematically evaluated against these safeguards. Specifically, we evaluate two low-cost attacks--abliteration and prefilling--that do not rely on gradient-based optimization. Across three harmfulness evaluation benchmarks (BeaverTails, HarmBench, and AdvBench), these attacks increase attack success rates against safeguarded open-weight models from below 10\% to a range of 16%-96%. To mitigate this vulnerability, we introduce abliteration-resistant tuning (ART), which incorporates an abliteration-based objective into training. ART can be layered onto existing defenses and reduces the success rates of abliteration, prefilling, and their combination by 10%-20%. These findings indicate that the attack surface for open-weight models is broader than previously characterized, and that evaluations of safeguarding defenses should incorporate a more diverse set of attack strategies beyond adversarial fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22741v3">GRADE: Graph Representation of LLM Agent Dependency and Execution</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 35 pages. Expanded analyses and revised figures. Code: https://github.com/yzhao062/grade
    </div>
    <details class="paper-abstract">
      Traces record what LLM agents did, leaving implicit what each step relied on. GRADE holds both in one graph; six published graph views are its projections. Execution edges come free; each costly dependency edge is graded as observed in the trace, declared by added logging, or inferred under an assumption, a schema we test one grade at a time. We recover the dependency layer from six tool-use, coding, and web corpora and measure its failure-prediction gain over run size. First, over a linear run-size baseline, it improves prediction on three corpora, but the gain depends on the learner and task sampling. On held-out corpora the layer stays above chance where run size does not, under one learner, with the same task-sampling caveat. Second, structure can pass for signal it does not carry: a layer inferred under full history restates run size. Within a corpus, a preregistered test does not separate the recovered layer from a degree-matched counterfeit that rewires each step's dependencies but keeps their number. The recovered layer's largest transfer lead over it reverses when the held-out corpus's task family is withheld. Before crediting structure, grade each edge, keep a run-size baseline, and score degree-matched attachment controls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07652v2">Bridging Natural Language and Interactive What-If Interfaces via LLM-Generated Declarative Specifications</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 34 pages 20 figures
    </div>
    <details class="paper-abstract">
      What-if analysis (WIA) lets users explore hypothetical scenarios by adjusting parameters, applying constraints, and scoping data through interactive interfaces. Current tools fall short: spreadsheet and BI tools require laborious setup, while LLM-generated interfaces frequently misinterpret analytical intent. It remains unclear whether an explicit intermediate representation is needed, or whether LLMs can generate WIA interfaces directly. We build a benchmark of 405 WIA questions across 11 types and 5 datasets with human-authored ground-truth intent, and find that direct one-shot generation across three LLMs fails to produce a working interface in 17-45% of cases and, when interfaces render, often misrepresents the question's objective (up to 77.5%) and constraints (up to 58.2%). We therefore present a two-stage workflow that translates NL questions into Praxa Specification Language (PSL) specifications, then compiles them into interfaces. Because errors localize to named properties, taxonomy-guided repair raises correctness from 52.42% to 80.42%, unlike errors in directly generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.02852v2">The Implications of Linguistic Illegibility for LLM Security</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      LLMs are trained to generate natural language. However, various strands of evidence indicate that an LLM's externalized linguistic outputs and mechanistically-extracted linguistic features can be an unreliable lens for understanding internal model computation. We introduce the term ``linguistic illegibility'' to broadly refer to scenarios in which an LLM's externalized or mechanistically-probed language artifacts fail to represent how the model actually thinks. We argue that the specter of linguistic illegibility is unavoidable for LLMs whose internal computations are not directly expressed via language, but rather math over activation spaces (with lossy translations between activation spaces and natural language happening at the bookends). If linguistic illegibility is always possible, then security mechanisms that rely on a model's linguistic self-reporting (e.g., chain-of-thought monitoring, constitutional self-critique, activation probing for linguistically-defined feature vectors) can never be completely sound; the model sandbox will always need isolation techniques whose guarantees do not depend on reading a model's linguistic state at all. We argue that observing a model's outputs using taint tracking is a promising approach for an effective sandbox: regardless of how a model linguistically self-reports, a taint tracking policy can define, a priori, various pieces of system state that should never be influenced by model-produced data. We also discuss several additional sandboxing mechanisms (e.g., robust virtualization, third-party auditing of sandboxing configurations) which collectively provide a critical floor beneath linguistic monitoring, and would have mitigated recent sandbox exploits by frontier models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09115v1">From Uncertainty to Action: Learning to Steer LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 22 pages, 9 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Steering an LLM agent means deciding whether to correct it, at which step, and with which mechanism. Uncertainty is often used to decide when to correct an agent, but whether it can guide these decisions remains unclear. We steer agent trajectories separately at every non-terminal step with each of four mechanisms and run each continuation to completion. The resulting stepwise outcome table (SOT) holds about 82,000 counterfactual continuations of 1,864 trajectories from three benchmarks and two agents. It shows that uncertainty can identify failing trajectories, but that no single signal reliably locates the step at which steering helps. We therefore propose VoS (Value of Steering), a trajectory-level monitor, offline or online, that learns from SOT the value of steering at each step and decides where to steer by it. A harm-budgeted trigger decides whether to steer, limiting the fraction of successful trajectories that VoS disturbs. VoS improves on unmodified execution in all 12 settings of benchmark, agent, and offline or online use, by 7.8 points on average, and outperforms the strongest of five existing uncertainty-triggered methods in 11, by 2.9 points on average. Ablations show that training on measured outcomes and a tight harm budget are both essential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05544v3">Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLM) and vision-language models (VLM) have achieved state-of-the-art performance, but they impose significant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this challenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimization and prove that a single uniform tolerance yields surrogate Pareto-optimal heterogeneous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value Decomposition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternating least-squares implementation. We apply PGSVD to both LLM and VLM, showing better accuracy at the same compression levels and inference speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09063v1">Multi-Label Topic Assignment via LLM Distillation: A Comparative Analysis of Generative vs. Discriminative Student Models</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Preprint. 9 pages
    </div>
    <details class="paper-abstract">
      Multi-label topic assignment for user-generated content (UGC) -- including product reviews and buyer-seller conversations -- poses unique scalability challenges in large-scale e-commerce due to informal language, extreme label sparsity, and rapidly evolving taxonomies. While utilizing Large Language Models (LLMs) as labeling oracles to distill ground-truth data has emerged as an industry standard to bypass prohibitive manual annotation costs, determining the optimal, low-latency architecture for the resulting student models remains an open challenge. To address this, we conduct a comprehensive evaluation across Small Language Model (SLM) parameter scales (1B, 4B, and 8B) and architectural paradigms (causal generative versus bidirectional discriminative). Comparing generative text-to-label classifiers against discriminative baselines (DeBERTa-V3 and ModernBERT), our analysis reveals a crucial data-dependent trade-off: while discriminative models outperform ultra-lightweight generative models on structured product reviews, even the smallest 1B generative model surpasses discriminative baselines on complex, multi-turn conversational data. Furthermore, generative models maintain robust performance under massive label-set expansion (up to 112 topics) and severe long-tail distributions, whereas discriminative baselines suffer a 35% drop in Macro-F1 at scale. Finally, we detail the successful production deployment of these optimized models across both product review and conversational domains, demonstrating strict latency compliance and tangible business impact at a global marketplace scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21347v4">Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 v4: expanded evaluation with the APEX-Agents benchmark and a GPT-5.4 cross-family judge. Revised analysis and appendices
    </div>
    <details class="paper-abstract">
      Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by implementing IG insights. Human experts using IG reports improve scaffold performance by 30.4 pp over the unmodified baseline scaffold, and coding agents leveraging IG-derived insights show consistent and stable gains. Across benchmarks, IG's scout-investigator architecture produces findings comparable in detection coverage to competing approaches, while domain experts rated IG reports as leading depth and evidence quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.27388v2">Handle with CARE: Can LLMs Reproduce How Online Communities React?</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as proxies for computational social analysis, yet faithfully representing the "thick descriptions" (Geertz, 1973) of human communities remains a critical challenge. Current evaluations often reduce social identity to static labels, sidelining how real-world groups navigate social shifts. To bridge this gap, we introduce CARE (Community-Aware Reaction Evaluation), a reaction-centered framework that benchmarks LLM-simulated discourse against the authentic, event-contingent responses of distinct communities to real-world news. Spanning 207 Reddit communities and covering 9,947 authentic reactions towards 2,166 news articles, CARE evaluates leading LLMs using a hierarchical taxonomy covering coarse attitudes and fine-grained communicative tones. Our empirical findings expose two critical failure modes in prevailing community-conditioning paradigms. First, while community context and targeted reasoning significantly enhance macro-level attitudinal and tonal profiling, these gains largely collapse at the instance level when predicting reactions to specific events. Second, the benefits of community conditioning are remarkably uneven: prompting strategies yield non-uniform shifts, where fidelity gains in some communities are offset by performance drops in others. This micro-macro divergence and community-level instability demonstrate that standard conditioning enables models to approximate static baseline profiles without capturing dynamic or equitable event reactions, establishing CARE as an essential diagnostic tool for community-aware social simulation. Our code and data are available at https://github.com/nuankw/Handle_with_CARE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09039v1">From High Recall to High Utility: Dataset-Adaptive Post-Processing of LLM-Generated Customer Intents</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Large language models can extract useful signals from heterogeneous enterprise data, but high-recall extraction often produces outputs that are duplicated, uneven in granularity, semantically overlapping, or too numerous for downstream systems and human reviewers to use effectively. We present a dataset-adaptive post-processing architecture developed for Customer Intent Extraction (CIE), where unstructured customer language is transformed into stable, traceable intent units. The approach separates recall-oriented extraction from utility-oriented reduction. Source-specific preprocessing first isolates evidence from multimodal plans, sparse operational records, and structured opportunity data. Candidate intents are then standardized and deduplicated, optionally enriched with metadata for embedding computation, represented in a shared semantic vector space, and grouped using a clustering strategy selected according to the candidate set's characteristics. Cluster-level keywords provide an explainability layer, while singleton reassignment requires agreement between embedding and keyword similarity. Finally, constrained language-model aggregation produces one concise intent per cluster without introducing unsupported concepts, and the resulting unit retains provenance, clustering, embedding, and generation metadata. This treats post-processing not as cosmetic cleanup, but as a semantic reduction layer converting high-recall LLM outputs into reusable enterprise intelligence. We also describe two downstream applications: Machine-Generated Intents, which infer likely objectives for customers lacking direct evidence from peer customers with similar profiles, and intent-guided semantic retrieval and mapping, which uses the stable intent as a query against a downstream decision space, illustrated here by mapping customer intents to business outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20021v2">Continuous Semantic Caching for Low-Cost LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted to ACM MobiHoc 2026
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly popular, caching responses so that they can be reused by users with semantically similar queries has become a vital strategy for reducing inference costs and latency. Existing caching frameworks have proposed to decide which query responses to cache by assuming a finite, known universe of discrete queries and learning their serving costs and arrival probabilities. As LLMs' pool of users and queries expands, however, such an assumption becomes increasingly untenable: real-world LLM queries reside in an infinite, continuous embedding space. In this paper, we establish the first rigorous theoretical framework for semantic LLM response caching in continuous query space under uncertainty. To bridge the gap between discrete optimization and continuous representation spaces, we introduce dynamic $ε$-net discretization coupled with Kernel Ridge Regression. This design enables the system to formally quantify estimation uncertainty and generalize partial feedback on LLM query costs across continuous semantic query neighborhoods. We develop both offline learning and online adaptive algorithms optimized to reduce switching costs incurred by changing the cached responses. We prove that our online algorithm achieves a sublinear regret bound against an optimal oracle, which reduces to existing bounds for discrete query models. Extensive empirical evaluations demonstrate that our framework approximates the continuous optimal cache well while also reducing computational and switching overhead compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.09008v1">Whose Memory Is It? Scope-Aware Commit Rules for Long-Term LLM Memory</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 20 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Persistent memory allows an LLM agent to carry experience across conversations, but it also turns a local reasoning mistake into a durable one. During deliberation, an agent may consider a plan, simulate a tool result, report another speaker's belief, and then reject all of them. If memory retains only the resulting sentences, those once-useful possibilities can later return as facts. The record is neither fabricated nor irrelevant; it has simply been detached from the context in which it was valid. We identify this missing context as \emph{discourse ownership}: the world, branch, or speaker that licenses a proposition. Our first finding is counterintuitive. Language models already carry a causally active signal for ownership, yet conventional memory interfaces discard it when they convert reasoning into records. We introduce CASK (Causally Anchored Scoping Keys), a commit rule that preserves this signal so that shared-world facts enter durable memory while provisional content remains available only within its original scope. Our second finding is that the most obvious way to preserve the signal---storing the discovered internal coordinates---is unreliable because equivalent representations need not keep the same coordinates. CASK instead preserves the stable relations that express ownership. Controlled long-conversation conflicts and tool-agent traces show that this design improves memory admission and prevents provisional content from contaminating later answers while complementing runtime provenance. The resulting commit boundary lets agents explore more possibilities without granting every intermediate sentence authority over future behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08994v1">Phoneme-Guided Initialization for LLM-based Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted at IEEE SLT 2026
    </div>
    <details class="paper-abstract">
      Speech large language models (speech LLMs) perform well on automatic speech recognition (ASR) when sufficient paired speech-text data is available, but their performance degrades in low-resource settings. A cascaded pipeline that performs speech-to-phoneme (S2P) conversion followed by phoneme-to-grapheme (P2G) conversion has been shown to outperform end-to-end speech LLMs in this regime, suggesting that phoneme-mediated processing is beneficial when paired data is scarce. We propose \textit{phoneme-guided initialization}, a simple method that uses this insight within an end-to-end framework: we pre-train the audio encoder on S2P and the LLM on P2G tasks, then connect them and fine-tune the full model end-to-end on the target ASR task. Experiments on Japanese (CSJ), Chinese (AISHELL-1), and two low-resource languages from Common Voice 25.0 (Tatar and Urdu) show that our method matches or outperforms both the cascaded S2P-P2G baseline and the end-to-end model without P2G initialization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.23264v3">Hidden in the Request: Explaining Unethical LLM Compliance through Token Relevance</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 SocialAgent, NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) are aligned to optimize for both helpfulness and harmlessness, these dual objectives may conflict, inevitably leading to alignment failures. This work systematically investigates instances where LLMs fail to exhibit ethical behavior. To understand the underlying mechanics of these vulnerabilities, we introduce a probing methodology that presents unethical scenarios to LLMs in three distinct structural modalities: objective classification tasks, subjective first-person statements, and direct requests for assistance. We find that model performance degrades in the request-for-assistance-based form. Using Layer-wise Relevance Propagation (LRP), we trace this discrepancy to an attribution bias: the model places greater emphasis on benign task-framing tokens (e.g., "Can you help me...") than on tokens signaling the underlying unethical behavior (e.g., "without getting caught"), which we term cue-tokens. We hypothesize that this under-attribution contributes to harmful compliance. To test this, we introduce two LRP-guided decoding methods that steer generation toward trajectories more relevant to cue tokens. Empirical evaluations show that these interventions promote safer responses, supporting cue-token attribution's role in compliance failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.29217v3">Advancing LLM-based phoneme-to-grapheme for multilingual speech recognition</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 NCMMSC 2026 camera-ready version
    </div>
    <details class="paper-abstract">
      Phoneme-based ASR factorizes recognition into speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G), enabling cross-lingual acoustic sharing while keeping language-specific orthography in a separate module. While large language models (LLMs) are promising for P2G, multilingual P2G remains challenging due to language-aware generation and severe cross-language data imbalance. We study multilingual LLM-based P2G on the ten-language CV-Lang10 benchmark. We examine robustness strategies that account for S2P uncertainty, including DANP and Simplified SKM (S-SKM). S-SKM is a Monte Carlo approximation that avoids CTC-based S2P probability weighting in P2G training. Robust training and low-resource oversampling reduce the average WER from 10.56% to 7.66%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09354v5">"Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted at CSCW 2026. 52 pages, 12 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Mental health is a growing global concern, prompting interest in AI-driven solutions to expand access to psychosocial support. Peer support, grounded in lived experience, offers a valuable complement to professional care. However, variability in training, effectiveness, and definitions raises concerns about quality, consistency, and safety. Large Language Models (LLMs) present new opportunities to enhance peer support interactions, particularly in real-time, text-based interactions. We present and evaluate an AI-supported system with an LLM-simulated distressed client (SimClient), context-sensitive LLM-generated suggestions (Suggestions), and real-time emotion visualisations. 2 mixed-methods studies with 12 peer supporters and 6 mental health professionals (i.e., experts) examined the system's effectiveness and implications for practice. Both groups recognised its potential to enhance training and improve interaction quality. However, we found a key tension emerged: while peer supporters engaged meaningfully, experts consistently flagged critical issues in peer supporter responses, such as missed distress cues and premature advice-giving. This misalignment highlights potential limitations in current peer support training, especially in emotionally charged contexts where safety and fidelity to best practices are essential. Our findings underscore the need for standardised, psychologically grounded training, especially as peer support scales globally. They also demonstrate how LLM-supported systems can scaffold this development--if designed with care and guided by expert oversight. This work contributes to emerging conversations on responsible AI integration in mental health and the evolving role of LLMs in augmenting peer-delivered care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08928v1">Multi-Aspect Runtime Verification for Simulation-Based V&V of LLM-Enabled Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 28 pages, 2 figures, 2 tables. Accepted at Modelling and Simulation for Autonomous Systems (MESAS 2026), to appear in Springer LNCS
    </div>
    <details class="paper-abstract">
      LLM-based agents are entering decision-support roles in defence staff work, where the obligations they must respect are already written down and binding, and where retraining is not available as a control because models arrive as procured components. What can be placed under engineering control is the interface between the agent and the systems it acts on. Those obligations are at once spatial, temporal and text-semantic, and a violation typically lives in the composition of a multi-step interaction, which is why per-event guardrails miss sequential tool-attack chains. We present a multi-aspect runtime-verification framework that decomposes a natural-language policy clause into a typed spatial/temporal/semantic triple over one canonical event stream, checks each aspect with its own monitoring specification, and fuses the verdicts through a four-valued algebra that carries provenance. The spatial aspect is interpreted over a weighted two-sorted location graph in which mission geometry and information-release topology are one object; we show that these spatial obligations are not in general subsumed by a first-order temporal specification. The past-time aspect runs on the unmodified MonPoly engine, which agrees with our reference monitor at every time point. Across two mission domains, casualty evacuation and contested sustainment, and one civil domain, composition under the precautionary blocking policy drives attack success to zero with no observed false positives and microsecond-scale per-event cost, while every single aspect and every pair leaves a substantial share of attacks succeeding. In a closed-loop experiment a policy-naive planner reaches a violating state in most unshielded missions and in none when shielded, and four refused episodes in five still recover to a compliant outcome.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08923v1">AdaGuard: Enhancing Safety and Policy Compliance with Reasoning-Enabled LLM-As-A-Judge Guardrails</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Enterprise generative AI applications require robust safety mechanisms that can accommodate diverse risk postures, evolving policies, and varying latency constraints. Current guardrail solutions often suffer from rigidity, relying on fixed policy sets and offering limited transparency or reasoning flexibility. We present Adaguard, an adaptive LLM-as-a-Judge framework designed to address these challenges through dynamic policy enforcement and adaptive reasoning-budget allocation. Built using supervised fine-tuning (SFT) and reinforcement learning (GRPO), AdaGuard generalizes to user-defined safety and compliance policies at runtime without requiring frequent model updates. A core innovation of our approach is the ability to dynamically infer the complexity of input-policy pairs, allowing the model to switch between high-speed black-box inference and explainable, reasoning-enabled moderation. This flexibility enables developers to balance stringent latency requirements with the need for actionable transparency. This adaptive capability allows AdaGuard to rival other guardrail and frontier models several times its size, while its auto-reasoning mode recovers the accuracy of always-on reasoning at a fraction of the latency
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08781v1">IdeaAnchor: Teaching LLMs to Turn Literature into Research Ideas</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Scientific research often begins by synthesizing ideas from a set of related papers to identify gaps and formulate new directions. However, training language models to perform this form of literature-grounded ideation remains challenging, as existing approaches based on prompting or feedback lack structured supervision for how papers should be synthesized. We introduce IdeaAnchor, a paradigm for training LLMs to perform research ideation using structured specifications as privileged signals. Each IdeaAnchor instance encodes how each input paper should be synthesized into a successful idea, including their functional roles, relationships, and target synthesis criteria. We build this paradigm by mining instances from published papers, capturing how real ideas emerge from prior literature. We then train models via demonstration, self-distillation, and reinforcement learning, and further enhance generation with retrieval at inference time. Experiments show consistent improvements in ideation quality. Our analysis reveals a functional decomposition: anchor-based training strengthens creative synthesis, retrieval enhances detail elaboration, and combining both yields the best performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08778v1">Sherpa: Teaching LLMs to Teach Adaptively</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 32 pages, 6 figures. Code and model are available at https://github.com/SALT-NLP/Sherpa
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly capable problem solvers, but being able to solve a problem is not the same as being able to teach it. Existing approaches to training LLMs as teachers rely on demonstrations, preference data, or predefined pedagogical criteria that specify what good teaching looks like. However, these signals are often not grounded in individual student learning outcomes, where effective teaching strategies can vary substantially across learners. To address this, we introduce Sherpa, a multi-turn reinforcement learning framework that instantiates multiple student archetypes with LLMs conditioned on distinct learning preferences and trains a teacher model to adapt its instruction by directly maximizing their learning outcomes. Teacher LLMs trained with Sherpa improve instructed students' performance across all archetypes by an average of 20.5 percentage points. Under MathTutorBench's evaluation, Sherpa raises the overall pedagogy score from 52.5% to 79.2%, indicating better teaching responses. Our human studies show that the trained teacher is preferred over the base model in 79.6% of pairwise comparisons. Together, Sherpa trains LLM teachers to adapt to diverse simulated students and become better aligned with human teachers, paving the road towards AI tutors teaching real students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08775v1">Agent in a Bottle: Can LLM Agents Turn Their Capabilities Into Cheap, Scalable Artifacts?</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can solve many narrow tasks, but querying them separately for millions of related instances can be prohibitively expensive. Can LLM agents autonomously create cheaper solutions for such workloads? We call this ability "bottling": the ability to turn general capabilities into task-specific solutions that balance answer quality and amortised cost. We introduce BOTTLED, a benchmark in which agents receive an entire unlabelled workload and must complete it under fixed time, compute and LLM API budgets. Agents choose their own approach, such as training a small model or writing a reusable program. Across ten models and three tasks, we find that strong zero-shot task performance does not reliably translate into strong bottling capabilities. Models with similar zero-shot scores can differ substantially after bottling, and 48 of 60 bottling runs score below the lower bound of the 95% confidence interval of their model's zero-shot performance. Moreover, 31 of 60 runs underperform the stronger of two small-model distillation baselines with the same token budget. Nevertheless, bottling can yield substantial savings: on query-product relevance classification, Opus 5 retains about 82% of its zero-shot macro-F1 at roughly 657 times lower reported cost. Bottling is also competitive with Jev, a "system one" model built especially for cheap, repetitive inference: Opus 5 on the same task recovers about 94% of Jev's macro-F1 at a quarter of Jev's projected full-workload cost. BOTTLED provides a basis for evaluating and improving agents' ability to invest limited resources in reusable solutions for large, repetitive workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.20740v2">Reinforcement Learning over Predictive Distributions for LLM Regression</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 27 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as flexible regressors capable of predicting real-valued quantities from heterogeneous inputs. Yet most LLM regression objectives optimize predictions independently, often yielding poor calibration. We introduce Distribution-Aware Reward (DAR), an on-policy reinforcement learning objective that instead jointly evaluates the empirical predictive distribution formed by multiple predictions for the same input. To translate this distribution-level objective into rollout-level rewards, we assign each prediction credit based on its leave-one-out contribution to the quality of the overall predictive distribution. This encourages predictions that are well-centered and appropriately dispersed around the target. We evaluate on three regression settings: a synthetic task probing interpolation and extrapolation, and two real-world scientific tasks involving code and molecular data. Across tasks, DAR produces better-calibrated uncertainty estimates while consistently reducing prediction error and improving ranking quality over supervised fine-tuning and pointwise reinforcement learning. Together, these results highlight the benefits of distribution-aware training for LLM regression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20658v2">Cooperative Profiles Predict Multi-Agent LLM Team Performance in AI for Science Workflows</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted at COLM 2026
    </div>
    <details class="paper-abstract">
      Multi-agent systems built from teams of large language models (LLMs) are increasingly deployed for collaborative scientific reasoning and problem-solving. These systems require agents to coordinate under shared constraints, such as GPUs or credit balances, where cooperative behavior matters. Behavioral economics provides a rich toolkit of games that isolate distinct cooperation mechanisms, yet it remains unknown whether a model's behavior in these stylized settings predicts its performance in realistic collaborative tasks. Here, we benchmark 41 open-weight LLMs across six behavioral economics games and show that game-derived cooperative profiles robustly predict downstream performance in AI-for-Science tasks, where teams of LLM agents collaboratively analyze data, build models, and produce scientific reports under shared budget constraints. Models that effectively coordinate in games and invest in multiplicative team production (rather than greedy strategies) produce better scientific reports across three outcomes, accuracy, quality, and completeness. These associations hold after controlling for multiple factors, indicating that cooperative disposition is a distinct, measurable property of LLMs not reducible to general ability. Our behavioral games framework thus offers a fast diagnostic for screening cooperative fitness before costly multi-agent deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08720v1">WorldSolver: Can LLM Agents Simulate the Physical Dynamics via Solver Generation?</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly advancing scientific and engineering problem solving, with physics simulation emerging as a challenging yet practical testbed for reproducing complex physical phenomena with application in embodied AI, games and films. As the workhorse of such simulation, a solver computes how the state of a dynamic system evolves over time. Building such solvers requires physical understanding to identify appropriate models, mathematical reasoning to formulate the underlying dynamics, and software engineering to implement them as executable code, yet this capability of LLM agents remains underexplored. To this end, we introduce WorldSolver, a benchmark of 168 simulation tasks derived from physical phenomena in 61 classic computer graphics papers, spanning 7 physical domains. Each task contains a code scaffold that provides a fixed simulation environment for the scene, with the solver implementation left for the agent to complete. Specifically, we evaluate them along three dimensions: Execution Checks for successful execution, Visual Fidelity for reproducing the intended dynamic behavior in the rendered simulation, and Physical Plausibility for physics-grounded verification of the generated dynamics. Experiments on frontier agents reveal that producing executable solvers is difficult itself, and satisfying visual and physical correctness is even harder. GPT-5.6-Sol and Claude-Opus-5 perform comparatively better than the other evaluated agents, yet achieve overall scores of only 48.7% and 46.7%, respectively. WorldSolver is an early step toward agentic solver generation, and we hope it helps drive progress toward agents that can faithfully simulate the dynamic physical world. Code is available at https://github.com/sirujiang/WorldSolver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08703v1">Agreement Is Not Validity: Cross-Model LLM Consensus in Diagnosing Student Failure Modes in K-12 Math Tutoring Dialogue</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Submitted to LAK27 as a short paper. Currently under review
    </div>
    <details class="paper-abstract">
      In K-12 mathematics tutoring, student-tutor dialogue provides rich evidence of learners' problem-solving processes and sources of difficulty. Learning analytics research increasingly relies on large language models (LLMs) to extract such information from dialogue for a variety of downstream tasks, including knowledge tracing, behavioral modeling, and diagnosis of student reasoning errors. However, the validity of these model-generated interpretations remains insufficiently understood. In this exploratory study, we examine the validity of LLM classifications of five student failure modes in mathematics tutoring dialogue using an operational diagnostic codebook: uncertainty, misattribution, operator selection, conceptual gap, and procedural slip. Across models, human-LLM agreement was moderate (kappa = .524-.597), while cross-model agreement was substantially higher (kappa = .755-.781; alpha = .769). These findings show that cross-model agreement can create a misleading appearance of correctness, challenging the assumption that consensus among LLMs constitutes evidence of valid learner interpretation. For learning analytics, the implication is clear: scalable labeling is useful only if the inferred constructs are valid, and model consensus cannot substitute for independent evidence of that validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.06729v2">Improving Diversity in LLM Short Story Generation</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate accurate responses, but these are void of diversity. We attempt to address this for the task of creative short story generation. Drawing on established writing conventions and known LLM limitations, we target variation in genre, tone, style, and named entities. To promote diversity across these dimensions, we introduce DivLM, an LLM post-training framework consisting of two phases. First, we perform continued pre-training on a creative writing corpus and restore instruction-following capabilities using weight residuals. We then apply reinforcement learning with a custom, composite reward function that jointly maximizes diversity across the targeted narrative dimensions while maintaining response quality. Our empirical results on two LLM families show that DivLM increases diversity metrics by more than 9% on average compared to alternative approaches, while preserving instruction following, overall response quality, and similarity to human outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08698v1">PrimitiveCAD: An LLM-Based Point-to-CAD Reconstruction with Primitive-Aware Tokenization and Operation Alignment</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Large-model-based point-to-CAD generation holds immense potential for advancing industrial design and enhancing 3D modeling efficiency. However, most existing methods approach the problem as a general point-cloud encoding and token prediction task, neglecting the tokenization and supervision specifically for CAD-related primitives. As a result, these methods often struggle to accurately reconstruct the intricate primitive structures. To address this limitation, we propose PrimitiveCAD, a novel multi-stage paradigm for point-to-CAD reconstruction that enhances the geometric accuracy of generated CAD models while better preserving critical geometric features. First, we introduce a primitive-aware point cloud tokenization model, enabling the system to learn more robust geometric representations from CAD point clouds. Next, we perform supervised finetuning on a large language model (LLM) and introduce an operation alignment loss to align key CAD operation frequencies, thereby improving the preservation of global shape features. Finally, we incorporate reinforcement learning (RL) and introduce a feature-line alignment reward to further reduce stochasticity and enhance the fine-grained preservation of geometric features. Experiments on the DeepCAD and Fusion360 datasets show that our method achieves state-of-the-art performance in code validity, geometric accuracy, and geometric feature preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02639v2">A Positive Case for Faithfulness: LLM Self-Explanations Help Predict Model Behavior</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      LLM self-explanations are often presented as a promising tool for AI oversight, yet their faithfulness to the model's true reasoning process is poorly understood. Existing faithfulness metrics have critical limitations, typically relying on identifying unfaithfulness via adversarial prompting or detecting reasoning errors. These methods overlook the predictive value of explanations. We introduce Normalized Simulatability Gain (NSG), a general and scalable metric based on the idea that a faithful explanation should allow an observer to learn a model's decision-making criteria, and thus better predict its behavior on related inputs. We evaluate 18 frontier proprietary and open-weight models, e.g., Gemini 3, GPT-5.2, and Claude 4.5, on 7,000 counterfactuals from popular datasets covering health, business, and ethics. We find self-explanations substantially improve prediction of model behavior (11-37% NSG). Self-explanations also provide more predictive information than explanations generated by external models, even when those models are stronger. This implies an advantage from self-knowledge that external explanation methods cannot replicate. Our approach also reveals that, across models, 5-15% of self-explanations are highly misleading. Despite their imperfections, we show a positive case for self-explanations: they encode information that helps predict model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08668v1">Semantic Behavioral Watermarking: Paraphrase-Robust and Forgery-Resistant Provenance for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Behavioral watermarking embeds an owner identifier in an LLM agent's high-level action choices, giving provenance without touching output tokens. Prior agent watermarks break in two ways. First, all three prior schemes bind the watermark to the exact action symbol, so renaming a tool desynchronizes decoding even when the observation is untouched; in AgentMark's own robustness test, paraphrasing the observation alone drops bit-recovery to 16.8%. Second, every prior agent watermark studies only removal: none asks whether an adversary can forge a trajectory that verifies as someone else's, a question answered affirmatively for text watermarks (Jovanović et al., 2024). We present Semantic Behavioral Watermarking (SBW): watermarking over semantic action clusters under history conditioning, with the public-cluster bin replaced by keyed collision-resistant binning whose fresh-bucket assignment is provably unpredictable in the random-oracle model. Across five agent models (3B-14B, four vendors) and three encoders the ordering holds on both benchmarks: on ToolBench (600 trajectories per model) detection under rewriting is 0.49-0.66 for cluster-level versus 0.05-0.17 for exact-symbol at a permutation-calibrated 1% FPR, at 72-83% choice agreement against 22-27% for logit biasing; on ALFWorld (100 episodes per model) it is 0.92-0.97 versus 0.00-0.01. Keyed binning takes adaptive forgery from 100% to the false-positive floor at the primary operating point (bge, r=64). We also mark the boundary that guarantee does not cover: when the adversary copies the victim's own steps, shuffled splicing is neutralized (0.000 on Qwen2.5-3B) but chained replay remains at 0.76-0.98 across the five models, reported as open. Paraphrase robustness costs about half of the per-step watermark capacity. Code is available at https://anonymous.4open.science/r/SBW-Agent-Watermark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04265v2">Strategic Evaluation of Planning Strategies for LLM Agents in Cyber-Physical Systems</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      LLM-agent evaluations commonly measure task success or agreement with a declared plan. In strategic cyber-physical systems, an architecture must also remain appropriate after autonomous participants respond and physics constrains outcomes. We introduce a controlled benchmark of planning-induced control trajectories: ordered planning operations and directives linking execution architecture to strategic response and physical consequences. Four coded executors (predefined, sequential, hierarchical, and search) control demand response for 40 prosumers on a radial feeder. The LLM declares or advises typed policies and mediates communication; schedules, base prosumer dynamics, stochastic actions, and power flow remain explicit code. Paired forced-mode counterfactuals, exact-prompt caching, common response draws with separate randomness streams, critic isolation, and event-level feasibility isolate comparisons. The Llama-3.3-70B experiments on this feeder distinguish three properties. First, forced search is the oracle in all five baseline seeds under the specified objective. Second, injected objective substitution preserves mode agreement at 1.0 while increasing cumulative voltage shortfall by 2.68x. Third, the 144-scenario, 576-episode factorial bank, using three repeated seeds, contains feasible oracles from predefined, sequential, and search. The prespecified stress-held-out ridge has mean regret 90.7 and no observed value over fixed sequential. A post-hoc constraint-aware analysis reduces regret to 29.0; a simple deadline rule attains 28.7, so this gain does not establish a learning advantage. An all-feasible ablation does not improve over fixed search. These are simulation-internal, descriptive comparisons. A five-model, 300-declaration extension tests interface behaviour, not cross-backbone physical rankings; shared-endpoint latency tails motivate probabilistic live feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08563v1">Adaptive Power Sampling for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 22 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Sequence-level power sampling has recently emerged as a training-free approach to reasoning by sampling from a sharpened output distribution of a base large language model (LLM). Nevertheless, existing methods typically sharpen the base model distribution uniformly across queries, overlooking variations in query difficulty and in how well the base model already handles each query. The goal of this work is to equip power sampling with query adaptivity. Theoretically, we show that the benefits of further sharpening are determined by the self-reward gap between correct and incorrect responses. Based on this insight, we propose \emph{Adaptive Power Sampling} (APS), which adjusts the sharpening exponent on a per-query basis at test time using the relationship between answer agreement and the model's self-reward. Experiments across diverse reasoning tasks, including MATH500, HumanEval, and GPQA, show that APS consistently outperforms power sampling with a fixed sharpening exponent, without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02800v2">BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates autoregressive generation by using a lightweight draft to propose multiple tokens for parallel verification. However, existing methods often require an additional draft model or weight representation, introducing non-negligible memory overhead on resource-constrained devices. Self-speculative approaches reduce this overhead, yet still face trade-offs between draft quality, target quality, and storage efficiency. We propose BitNest, a bit-nested speculative decoding framework that embeds a low-precision draft directly into the higher-precision target representation. Instead of deriving a draft from a predefined target, BitNest first constructs a strong low-precision base and then recovers the higher-precision target through residual refinement, enabling both models to share a single physical weight representation. BitNest further extends this progressive-precision design to the KV cache for long-context inference. Across multiple 7B--8B edge-friendly LLMs and diverse workloads, BitNest achieves an average speculative acceptance rate of 95.2% while closely preserving higher-precision model quality, and delivers 1.48--1.61x end-to-end speedup over FP16 autoregressive decoding. On the LLaMA models supported by all representative self-speculative baselines, BitNest also achieves consistently competitive or higher decoding speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26189v2">VietBinoculars: A Zero-Shot Approach for Detecting Vietnamese LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 39 pages
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models has intensified the challenge of distinguishing LLM-generated text from human writing in non-English languages. This study introduces VietBinoculars, a zero-shot detection framework coupling PhoGPT-4B observer and performer models with calibrated global decision thresholds. By utilizing specialized Vietnamese BPE tokenization, the method eliminates byte-level fragmentation and probability dilution common in massive multilingual backbones. Evaluated across multi-domain benchmarks, VietBinoculars achieves an area under the ROC curve exceeding 0.99. Under optimal Youden's J thresholds and greedy decoding, detection accuracy reaches at least 98.78\%, while significantly outperforming baseline Binoculars, zero-shot detectors, and commercial tools on creative Capybara prompts. Even under a strict false positive rate constraint of 0.06\%, the detector maintains F1-scores between 83.15\% and 94.70\%. Detection performance consistently improves with sequence length, stabilizing at optimal accuracy for passages containing 450 to 550 tokens. Extended stress testing across 48 distinct model-decoding configurations and three post-generation rewriting strategies delineates practical operational boundaries. VietBinoculars exhibits robust resilience against single-pass paraphrasing and human-style revisions, but experiences notable performance degradation under high-entropy sampling and iterative double paraphrasing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08406v1">Case-Level Verification in Scanner-LLM Cascades: Overcoming the Alert Aggregation Bottleneck to Expand the FRR-TPR Trade-off Space</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted at the 3rd International Conference on Intelligent Computing and Data Analysis (ICDA 2026). To appear in the ACM International Conference Proceedings Series (ICPS). 17 pages
    </div>
    <details class="paper-abstract">
      Dynamic Application Security Testing (DAST) scanners achieve high recall but also produce a large number of false positives, resulting in substantial manual triage costs. Large Language Models (LLMs), when used for independent detection, achieve extremely high recall (95.4%-100%) but also exhibit prohibitively high false positive rates (49.6%-85.0%), precluding their use as standalone replacements for scanners. A natural solution is a two-stage cascade consisting of scanner detection followed by LLM verification. However, a verification-granularity issue that has long been overlooked in practice creates a structural bottleneck: alert aggregation binds multiple true and false cases into a shared decision unit, such that removing a false positive inevitably eliminates true positives aggregated within the same alert group. This creates a trade-off bottleneck between the False-positive Reduction Rate (FRR) and the True-positive Rate (TPR). We formalize this bottleneck by showing that the alert-level false-positive set is a subset of the case-level false-positive set, and introduce a Case-Level, per-case verification strategy that shifts the decision granularity from the alert level to the instance level, independently replaying HTTP requests and making an independent determination for each detected case. Evaluation on the dual testbeds of Damn Vulnerable Web Application (DVWA) and WebGoat shows that the empirically best Alert-Level operating point achieves FRR=42.86% (TPR=51.7%). The Case-Level Baseline achieves FRR=47.6%, an improvement of 4.7 percentage points (+4.7 pp), while the Case-Focused Evidence Verification Prompt (CEV-Prompt) increases TPR from 55.2% to 62.1% at the same FRR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08405v1">Learning from Failures: A Failure-Driven Prompt Refinement for LLM-Based Vulnerability Analysis</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted at CSoNet 2026. 15 pages, 6 figures/tables combined
    </div>
    <details class="paper-abstract">
      Large Language Models have emerged as promising tools for software vulnerability analysis, but their effectiveness depends heavily on prompt design. Existing research primarily compares prompting strategies using aggregate performance metrics, providing limited insight into why models fail or how prompts can be improved systematically. We propose Failure-Driven Prompt Refinement (FDPR), a methodology that analyzes recurring model failures to guide evidence-based prompt refinement. Using the Damn Vulnerable Java Application (DVJA), we identify recurring failure modes, including false positives, false negatives, unsupported reasoning, and CWE misclassification, and translate them into targeted prompt refinements. We then evaluate the resulting prompt on the Juliet Test Suite and perform cross-model validation to assess generalizability. The results show that failure-driven refinement improves the reliability of LLM-based vulnerability analysis while yielding reusable prompt design principles. More broadly, this work demonstrates that recurring model failures provide a principled foundation for prompt engineering, enabling the systematic development of more reliable LLM-based vulnerability analysis systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08402v1">VETTA: Coordinating Turn- and Token-Level Credit Assignment for Multi-Turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Multi-turn LLM agents often receive sparse task feedback across several interactions, while generating each response token by token. This creates two related credit-assignment questions: which responses helped achieve the outcome, and which generation decisions mattered within each response? Existing methods typically focus on only one level: turn-level methods evaluate complete responses but do not distinguish the decisions within them; token-level methods can propagate feedback across turns but do not explicitly model credit for each response. These complementary limitations motivate learning credit at both levels and coordinating it in a single policy update. We introduce VETTA, a credit assignment method that jointly learns turn- and token-level values through separate heads on a shared lightweight critic. VETTA computes advantages along both temporal sequences and combines each turn advantage with a within-response-centered token residual for PPO updates. Furthermore, to reduce value-learning cost, the critic retains only early Transformer blocks from the pretrained checkpoint used to initialize the actor. On two challenging agent benchmarks, ALFWorld and WebShop, VETTA improves success rates over PPO by 37.5% and 22.3%, respectively, with Qwen2.5-1.5B-Instruct and achieves success rates of 95.5% and 76.0%, respectively, with Qwen2.5-7B-Instruct. Critic-depth comparisons further show strong task performance with substantially lower critic-side computation. These results suggest that a compact shared critic can coordinate turn- and token-level credit to improve agent performance while keeping value estimation efficient. Code is available at https://github.com/Jiaju-Chen/VETTA-official.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21075v4">Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Related work updated
    </div>
    <details class="paper-abstract">
      Engineering LLMs to accelerate life sciences research requires a robust alignment with biomedical knowledge. We observe that biomedical text exhibits a fundamentally different uncertainty structure from general text: dense low-confidence runs encode epistemic knowledge gaps (dense causal chains, rare entities) rather than the sparse aleatoric stylistic variation typical of general text. Based on this discovery, we propose Balanced Fine-Tuning (BFT), a dual-scale post-training method that combines group-normalized token reweighting with sequence-level reallocation toward knowledge-dense samples exhibiting dense epistemic uncertainty. Across medical evaluation, biological reasoning, sparse-reward RL, and biological representation tasks, BFT provides more consistent gains than SFT and DFT under a shared training setup. When replacing the default closed-source backbones in GeneAgent (GPT-4o) and VCWorld (Gemini-2.5-Flash), the BFT-aligned 70B model delivers stronger performance across biological process reasoning and chemical perturbation prediction. Critically, all BFT variants further improve after subsequent GRPO with sparse rewards, while SFT and DFT degrade, suggesting that epistemic-aware post-training provides a more robust policy initialization. Beyond text generation, BFT-aligned LLMs produce more accurate and professional biomedical profile texts; after encoding these profiles with a text embedding model, the resulting representations support gene-level, cell-level, and perturbation-response tasks, suggesting that BFT-enhanced generation can facilitate biological representation and, in turn, broader biomedical downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08373v1">ECO: Energy-Oriented Configuration Optimization for Attention FFN Disaggregated LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Energy-efficient LLM serving requires minimizing serving GPU energy while meeting latency and throughput service-level objectives (SLOs). Attention--FFN disaggregation (AFD) enables separate resource allocation and operating controls for attention and expert computation, but their energy effects remain coupled through the execution pipeline. Realizing its energy-saving potential therefore requires navigating a hierarchical configuration space in which deployment structures constrain admissible controls and shape their end-to-end effects. Finding low-energy configurations that meet SLOs is challenging because physical evaluations are costly and only a small fraction of candidates can be measured. We present Energy-Oriented Configuration Optimization (ECO), which jointly searches deployment structures and their admissible operating controls under a limited measurement budget. ECO constructs a structure-aware energy prior from calibrated stage behavior and pipeline dependencies, then learns residual prediction errors with a Gaussian process. Its cost-aware constrained Bayesian optimization prioritizes measurements according to expected energy improvement while accounting for SLO feasibility, execution success, and evaluation cost, and returns the lowest-energy measured feasible configuration. Across all 16 scenarios on A6000 and A100 with Qwen and DeepSeek, ECO's frozen configurations, evaluated on disjoint requests, reduce serving energy by 40.5\% and increase output token rate by 20.7\% on average relative to baselines while meeting target SLOs. Across the 8 A6000 scenarios, its selected feasible energy averages 33.1\% below generic constrained Bayesian optimization and 25.8\% below genetic search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.37858v2">Storage Is Not Strategy: State-Conditioned Support Control for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Many localized large language model (LLM) unlearning methods select a small parameter subset from a localization signal and keep it fixed during optimization. The parameters most associated with a target, however, need not be the best ones to update, and candidate interventions can change value as optimization proceeds. In a controlled experiment, a storage-localization score reaches an area under the receiver operating characteristic curve (AUROC) of 0.981, yet storage identity agrees with the better intervention on only 17/36 targets, while low-rank adaptation (LoRA) wins 35/36. We introduce Intervention Score, which ranks editable groups by the predicted effect of the actual unlearning update while accounting for collateral damage, and use it to form the static intervention-value baseline (Static-IV). We then introduce selective dynamic intervention re-ranking (DIR-R), which revisits that subset only when a calibrated probe justifies the comparison. On the Natural-TOFU dataset, our method has positive descriptive margins in 19/20 comparisons between methods and objectives, although several are near zero. On the LACUNA localization-precision benchmark, our mean terminal utility is higher in all six negative preference optimization (NPO) and SimNPO comparisons: NPO margins range from +0.431 to +0.848, and SimNPO margins range from +0.503 to +0.571. The gradient-difference (GradDiff) objective reveals substantial field dependence. Relative to Static-IV, the primary four-field GradDiff evaluation has six wins, six ties, and no losses, with mean and median paired gains of +0.165 and +0.0025. The evidence supports separating localization, initial intervention selection, and checkpoint-dependent support revision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14345v6">PAC-CF: Calibrating Irreversible Frontier Pruning in LLM-Guided Search</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 26 pages. Major revision. Earlier versions circulated under the title PAC-MCTS and reported controlled proof-of-concept experiments. This version introduces native-trajectory conformal calibration, frozen-margin deployment, controller-agnostic integration, and benchmark-based multi-domain evaluation
    </div>
    <details class="paper-abstract">
      LLM-guided search explores multiple candidate trajectories, but at substantial test-time cost. Pruning low-scoring frontier candidates can control this cost, yet it also turns potentially biased evaluator scores into irreversible decisions: systematic ranking errors can persist under repeated scoring and remove useful branches. We propose Probably Approximately Correct Conformal Filtering (PAC-CF). Its fixed-frontier analysis formulates elimination as an $(\varepsilon,δ)$-PAC problem under bounded evaluator bias; its operational rule separately calibrates a score-gap threshold on held-out tasks by running the original controller without PAC-CF and using post-search verifier labels to measure the deficit of solution-preserving candidates relative to the frontier leader. Conditional on exchangeable native-controller tasks with nonempty protected exposure, conformal calibration gives finite-sample coverage for retaining at least one verifier-defined valid continuation at every protected frontier on the native trajectory. At deployment, PAC-CF removes only candidates whose gap from the highest frontier score exceeds the frozen threshold. We evaluate PAC-CF across three domains, five controllers, and four request budgets from B100 to B500. In the cross-domain/controller macro averages, the point estimates for all three workload measures are lower at every budget; the paired-bootstrap 95\% confidence interval for utility excludes zero at B100 and B200. For pruning-aware ToolTree, the full-test-set cross-domain utility difference is $+4.38$ points at each tested budget; on the natural-termination sensitivity cohort, physical requests decrease by $18.94$--$18.95\%$ and end-to-end token usage by $23.57$--$23.76\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08330v1">MoF: Preference-Aware Mixture Modeling for Black-Box LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted at EMNLP 2026
    </div>
    <details class="paper-abstract">
      Proprietary Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet aligning their outputs with diverse user preferences remains challenging. Existing personalization approaches for black-box LLMs often rely on user-specific scoring heads, causing the number of personalized parameters to grow linearly with the number of users and requiring additional adaptation for unseen users. To address these limitations, we propose Mixture-of-Facets (MoF), a scalable personalization framework for black-box LLMs that models user preferences as compositions of shared latent preference facets rather than dedicated user-specific parameters. MoF performs personalization through history-conditioned routing over shared facet heads, enabling personalization for users unseen during training without additional parameter updates. Across diverse personalization tasks, MoF delivers stronger personalization performance while maintaining a more scalable and parameter-efficient design than prior approaches. Additional analysis indicates strong generalization to unseen users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08328v1">Enhancing LLMs with Cognitive-Affective Personality Inference for Simulating Human Social-Psychological Behavior</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted at NeurIPS 2026
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to simulate human participants in social and behavioral studies, yet static persona prompting typically maps a participant profile and an experimental scenario directly to a response, entangling stable dispositions with situation-specific interpretations. To address this limitation, we introduce \textbf{SPIN}, a cognitive-affective personality system-inspired inference pipeline for simulating human social-psychological behavior. Specifically, SPIN implements this structured inference process through three zero-shot LLM calls that compile a task-blind participant core, elicit condition-specific cognitive-affective states, and read out decisions from those states, thereby reusing stable personality structure while routing each trial-specific response through an explicit state representation. We evaluate SPIN on two reconstructed social-psychological study families spanning uncertainty reasoning and pluralistic ignorance, across four base LLMs. Compared with blank, demographic, narrative, and chain-of-thought prompt variants, SPIN consistently delivers the strongest overall alignment performance across base LLMs and study families. Ablations and state analyses further show that both personality compilation and structured state elicitation contribute to the gains, and that the elicited states shift interpretably across informational and normative conditions. These results suggest that structured personality-state inference can improve benchmark-level behavioral alignment beyond richer persona descriptions or generic multi-step reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08312v1">CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted to EMNLP 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      Continual learning (CL) is essential for Large Language Models (LLMs) to sequentially adapt to evolving tasks. To mitigate catastrophic forgetting, recent advances implement low-rank adaptation with orthogonal projections (e.g., O-LoRA) to isolate task parameters. However, we reveal that such strict geometric constraints trigger an "Orthogonality Dilemma": rigid parameter isolation impedes the transfer and accumulation of shared representations across semantically related tasks. In this work, we propose a new replay-free method, called Consolidation and Decoupling LoRA (CoDe-LoRA), for CL of LLMs. CoDe-LoRA disentangles the learning process into Consolidating Universal Knowledge and Decoupling Task-Specific Knowledge. To achieve this, CoDe-LoRA leverages an adaptive null space projection mechanism and semantic routing to balance knowledge accumulation with task-specific adaptation. Experimental results across four backbones and three CL benchmarks show that CoDe-LoRA achieves the best average accuracy. Our code is available at https://github.com/Estrellajer/CoDe-LoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.19979v4">Enhancing High-order Interaction Awareness in LLM-based Recommender Model</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Long paper accepted to EMNLP 2024 Main. 16 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated prominent reasoning capabilities in recommendation tasks by transforming them into text-generation tasks. However, existing approaches either disregard or ineffectively model the user-item high-order interactions. To this end, this paper presents an enhanced LLM-based recommender (ELMRec). We enhance whole-word embeddings to substantially enhance LLMs' interpretation of graph-constructed interactions for recommendations, without requiring graph pre-training. This finding may inspire endeavors to incorporate rich knowledge graphs into LLM-based recommenders via whole-word embedding. We also found that LLMs often recommend items based on users' earlier interactions rather than recent ones, and present a reranking solution. Our ELMRec outperforms state-of-the-art (SOTA) methods in both direct and sequential recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08268v1">DySCo: Dynamic Sharding for Collaborative Edge-Cloud LLM Inference with Depth-Synchronized Batching</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 article under submission
    </div>
    <details class="paper-abstract">
      Pervasive intelligent applications are increasingly deployed on mobile and Internet of Things (IoT) edge devices. Consequently, Large Language Models (LLMs) are increasingly used to support these applications. Yet, due to their high resource demands, LLMs are mostly deployed in the cloud. Layer-wise edge-cloud inference lets resource-constrained edge devices contribute computation to LLMs they cannot host in full. However, heterogeneous split points introduce two coupled inefficiencies. First, edge execution and communication create idle gaps between cloud invocations. Second, requests arriving at different model depths cannot be conventionally batched. We present DySCo, a collaborative runtime that keeps KV caches local and introduces dyForward, a model-aware layer-range executor that runs configurable contiguous layer ranges from resident model shards without reloading weights. For multi-edge serving settings, we introduce depth-synchronized batching (DSB), which advances heterogeneous requests to the deepest cut and batches their common suffix. Experiments across heterogeneous devices, two model families, and local and wide-area links show that idle gaps increase the latency of subsequent GPU forward calls even when waiting time is excluded, adding up to 25 ms of additional cloud-side suffix latency per decoding step in our measurements. At an average concurrency of eight, DSB improves throughput by 275% over FIFO, 48% over exact-match batching, and 79% over round-robin interleaving while reducing mean per-session latency. Together, these results show that requests with different edge-cloud splits can reuse resident cloud weights and share batched suffix computation. The artifact repository for this work is publicly available at: https://github.com/Large-scale-Sustainable-Computing-LSC/dysco-artifact
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23184v2">Unbiased Reward Modeling from Implicit Feedback for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Despite the success of reinforcement learning from human feedback (RLHF), existing reward modeling methods largely rely on explicit feedback, which is costly to collect and difficult to scale. This work studies implicit reward modeling, learning reward models from implicit user feedback, such as clicks, copies and skips. While scalable and cost-effective, implicit feedback poses two key challenges: It lacks definitive negative samples, which makes standard positive-negative classification methods inapplicable; It suffers from selection bias, where responses have heterogeneous propensities to elicit feedback, which further obscures definitive negative samples. To address these challenges, we propose ImplicitRM, which learns unbiased reward models from implicit feedback. It stratifies training samples into four latent groups using a stratification model and derives a likelihood-maximization objective that is theoretically unbiased, thereby addressing both challenges. Experiments across diverse LLM backbones and benchmark datasets validate that ImplicitRM learns accurate reward models from implicit feedback and improves performance on downstream RLHF tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08246v1">LeanPlan: Optimal Planning with LLM-Generated Heuristics and Admissibility Proofs</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Frontier large language models (LLMs) can generate heuristic functions that guide search to achieve state-of-the-art performance in satisficing planning, where any plan is acceptable. However, these heuristics are not guaranteed to be admissible and can lead to suboptimal plans. We introduce LeanPlan, the first planning system that finds optimal plans with LLM-generated heuristics whose admissibility is machine-checked. Given a domain description and training tasks, an agentic loop uses planner feedback to iteratively improve a reusable domain-specific heuristic, its admissibility proof and the required domain assumptions. LeanPlan implements the heuristic, its proof and an efficient planner with machine-checked grounding and search in Lean 4. We evaluate LeanPlan on ten domains from the International Planning Competition and three new domains, using test tasks with up to 57 times as many objects as the training tasks. With GPT-5.6 Sol in the agentic loop, we successfully generate heuristics and admissibility proofs for all these domains. With the resulting heuristics, LeanPlan usually expands fewer states than the state-of-the-art Scorpion planner and solves more tasks overall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02199v1">TACO: Ternary Absolute-max Column-wise One-sparse Optimizer for LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-10-01
      | 💬 24 pages, 7 figures, 10 tables. Code available at https://github.com/Jichao2357/TACO_optimizer
    </div>
    <details class="paper-abstract">
      Full-parameter fine-tuning of large language models (LLMs) incurs substantial optimizer state memory overhead, limiting the model sizes that fit on modern GPUs. Existing approaches either compress optimizer state, abandon first-order gradients, or change the update geometry while retaining dense state. The recently introduced Muon optimizer reduces optimizer memory through matrix-valued updates. Still, its geometry differs from AdamW and can lead to performance degradation when fine-tuning AdamW-pretrained models. To reduce optimizer memory without sacrificing accuracy or computational efficiency in LLM fine-tuning, we propose Ternary Absolute-max Column-wise One-sparse optimizer, or TACO, which follows Muon's operator-norm steepest-descent view but takes the geometric route further. TACO computes the exact steepest-descent direction under a dimension-normalized $1\to1$ operator norm by selecting the sign of the largest magnitude entry in each column of two-dimensional weight matrices. This retains first-order gradients while making optimizer state memory nearly negligible. Our practical TACO optimizer maintains only a small set of low precision gradient components per column, reducing persistent optimizer state by $174\times$ relative to AdamW8bit (from 27.7 GB to 0.16 GB) and peak training memory by $2.9\times$ (from 80.6 GB to 27.5 GB) on OPT-13B, while achieving comparable accuracy and runtime. TACO further enables full-parameter fine-tuning of 30-32B-parameter models on a single 80 GB H100 GPU across multiple model families and tasks.
    </details>
</div>
