# Market & Startup Landscape: Human Motion / Demonstration Data for Robot Learning (as of 2026-10-16)

Note on sources: Many pricing and "landscape" pages are vendor blogs (DexSet, Flikforge, Labellerr, Truelabel, Claru, Unidata), so their figures are self-reported. TechCrunch, company pages (figure.ai), and arXiv are the strongest sources used here. All dates refer to 2026 unless stated.

## 1. Who is collecting human video / motion data for robots, and with what sensors?

### Takeaway
By late 2026 human-demonstration data is a crowded, well-funded supply chain. Humanoid OEMs (Figure, Tesla) run their own crowdsourced or first-person video pipelines. Specialist vendors (Mecka AI, XDOF, Human Archive, DeepReach, Scale AI, Micro1, Objectways, Encord, Innodata) collect egocentric video, teleop and mocap. Sensors range from plain smartphones and head-mounted caps to RGB-D headsets, tactile gloves, full-body mocap suits and Vicon optical rigs.

### Cited Findings
**Humanoid / foundation-model companies**
- Figure AI launched **Index** (formerly stealth "Project Go-Big") on Aug 25, 2026. It is an iOS/Android two-sided marketplace where "creators" record household and workplace tasks for cash bounties, or book gig workers who film first-person footage on-site. Reported numbers: 16M video uploads, about 30 min of footage ingested per second, 264K downloads, 108 countries, 44K+ weekly active users, $15M paid to contributors, and $1B+ committed to data collection and compute over the next 12 months. Diversity per 1,000 hours: 373 tasks, 1,146 objects, 116 environments. Capture is mainly smartphone video, with some head-mounted rigs. — [Humanoids Daily](https://www.humanoidsdaily.com/news/figure-ai-unveils-index-crowdsourcing-real-world-human-video-to-train-helix)
- Figure says its Helix model learned to navigate cluttered spaces from egocentric human video alone, with zero robot demos. The video was collected passively in Brookfield homes (Brookfield owns 100K+ residential units). — [Figure: Project Go-Big](https://www.figure.ai/news/project-go-big)
- Tesla says Optimus learned two-handed household chores mostly from first-person human demonstration video rather than teleop, all in one multitask network. The evidence is company demo videos (Oct 2026). — [Startup Fortune](https://startupfortune.com/tesla-says-optimus-learned-household-chores-just-by-watching-video/)
- Physical Intelligence reportedly found that fine-tuning the full pretrained π0.5 on human video "can double the performance" on tasks shown in the videos, and that human-to-robot transfer emerges at scale. This comes from social posts, not a paper. — [Texxr](https://texxr.com/893927/physical-intelligence-improves-robot-vision-with-human-video)
- NVIDIA GR00T uses a "data pyramid": teleop robot data at the top, synthetic data in the middle, and web plus human egocentric video at the base. The latest release reportedly includes 20,000 hours of curated egocentric human video. This is a vendor-blog claim. — [Flikforge](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/)
- Dyna Robotics: Dyna-2 task success reportedly rose from 20% to 53% (Aug 2026) after its training set grew from 1,000 hours to 1M+ hours of egocentric footage. The source is a secondary outlet with no primary link. — [Crypto Briefing](https://cryptobriefing.com/robotics-labs-motion-video-data-ai/)
- Agile Robots (Munich, "Robot Academy", ~Oct 8, 2026) uses hand tracking to turn ordinary worker videos into trajectories and skips teleop rigs. — [Crypto Briefing](https://cryptobriefing.com/robotics-labs-motion-video-data-ai/)
- Genesis AI pairs its dexterous hand with a data glove that maps 1:1 to the human and robot hand. It also uses head-camera egocentric video and internet video (May 2026). — [Robotics Tomorrow](https://www.roboticstomorrow.com/news/2026/05/06/genesis-ai-unveils-gene-265-the-first-ai-brain-to-enable-robots-with-human-level-physical-manipulation-capabilities/26519/)
- Apple's EgoDex dataset (2025): 829 hours of 1080p/30Hz egocentric video, 338K episodes, 194 tasks, ~90M frames. It was captured with ARKit on Apple Vision Pro using bare hands, with per-joint hand and upper-body pose labels. License is CC-BY-NC-ND, so it cannot be used commercially. — [arXiv 2505.11709](https://arxiv.org/pdf/2505.11709); [GitHub](https://github.com/apple/ml-egodex)

**Data vendors / startups**
- **Mecka AI** (founded 2024) pays people to record everyday tasks (making coffee, fixing cars) with body sensors and smartphones. Its pitch is to be the "Scale AI / Mercor / Surge for robotics." — [TechCrunch, Oct 7 2026](https://techcrunch.com/2026/10/07/robot-data-startup-mecka-ai-nabs-60m-from-sequoia/)
- **XDOF** (UC Berkeley spinout; CEO Philipp Wu, CTO Fred Shentu) sells teleop and egocentric data as an "outsourced data supply chain." It builds on GELLO, a low-cost teleop system whose device BOM is reportedly under $300. Its ABC-130K dataset has 3,553 hours of bimanual manipulation, all from one workstation type, which limits background diversity. — [Dealroom](https://dealroom.co/news/149061-xdof-nears-1-2b-series-b-just-months-after-70m-series-a/); [Pebblous](https://blog.pebblous.ai/blog/xdof-teleoperation-data-collection-rig/en/)
- **Human Archive** (YC; founders from Berkeley/Stanford) has Indian gig workers at home-services, hotel and restaurant partners wear camera caps. Its rigs include headset RGB-D, force feedback, full-body mocap suits, tactile gloves, and synchronized chest and wrist cameras. It has 1,000+ active headsets and 7+ hardware products, and started on iPhones before building custom rigs. It is expanding to Southeast Asia and the US and building an open platform where anyone can contribute and earn. — [TechCrunch, May 26 2026](https://techcrunch.com/2026/05/26/human-archive-taps-into-indias-services-startups-to-collect-data-for-physical-ai/)
- **DeepReach** (YC) says its data is "in production with multiple frontier model and robotics companies," scaling to 10,000+ devices in 50 countries. — [YC robotics directory](https://www.ycombinator.com/companies/industry/robotics)
- **Egolab.AI** (India) collects egocentric video from factory workers wearing lightweight head cameras. — [Egolab coverage via search summary](https://startupfeed.in/egocentric-data-india-robot-training-boom/)
- **Scale AI** extended its autonomous-vehicle data operation into robotics: a collector network, data-collection robots, human demo capture, and "data factories and robot gyms" in Mexico, China and Indonesia. Named robotics customers include Physical Intelligence, Generalist and Cobot. — [Truelabel](https://truelabel.ai/blog/best-egocentric-video-data-providers); [Crypto Briefing](https://cryptobriefing.com/robotics-labs-motion-video-data-ai/)
- **Micro1** is expanding from LLM data into robotics with gig workers in 50+ countries using head cams. — [TechCrunch](https://techcrunch.com/2026/10/07/robot-data-startup-mecka-ai-nabs-60m-from-sequoia/); [Labellerr](https://www.labellerr.com/blog/top-egocentric-data-providers-robotics/)
- **Encord** is primarily annotation tooling (SAM 2 segmentation and tracking). It also collects its own egocentric data in factories and is trialing EEG brain-wave-tagged data with Zander (Jul 2026). — [TechCrunch, Jul 26 2026](https://techcrunch.com/2026/07/26/are-brain-waves-the-next-unlock-for-physical-ai/)
- **Objectways** is a full-pipeline capture-to-label services firm that publishes fewer scale figures than rivals. — [Truelabel](https://truelabel.ai/alternatives/objectways)
- **Innodata** opened a Vicon optical mocap lab in Ridgefield Park, NJ (Sep 30 2026). — [Crypto Briefing](https://cryptobriefing.com/robotics-labs-motion-video-data-ai/)
- **Tacta Systems** makes skill-capture gloves (force, motion, temperature, plus video), has a Singapore production facility, and plans customer deployments in early 2027. — [TrendHunter](https://www.trendhunter.com/trends/tacta-systems-opens)
- **Flikforge** sells curated catalogs: EmbodiedAI-500k (500K hrs of activity video) and EmbodiedAI-75k (1,000+ primitive skills), delivered in MCAP, LeRobot and RLDS formats with provenance metadata. — [Flikforge](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/)

**Hardware reference costs:** ALOHA rig ~$20K, Mobile ALOHA ~$32K, UMI gripper under $1K, egocentric headset kits $300–$3,500. — [DexSet, Jul 2 2026](https://dexset.ai/blogs/robot-training-data-costs-pricing-complete-2026/)

### Inferences
- Collection is moving away from expensive teleop rigs toward cheap first-person capture on phones and caps, plus software that lifts video into hand/body trajectories (Agile Robots, Figure Index). A webcam/monocular pose tool fits this software layer.
- The big players (Figure, Scale, Mecka) compete on scale of collector networks. A small startup cannot win on raw hours. It can only win on tooling, quality, or niche domains.

### Gaps
- I found no 2026 primary sources specific to Skild AI, 1X, Sunday Robotics, Generalist AI, Google DeepMind, or Meta Aria Gen 2 / Ego-Exo4D data collection; searches returned nothing usable.
- NVIDIA's "20,000 hours" GR00T figure comes only from a vendor blog. It needs checking against NVIDIA's own release.

## 2. Funding rounds (2024–2026) and pricing per hour

### Takeaway
Robot data is now a venture category of its own. Mecka AI raised a $60M Series B (Sequoia, Oct 2026) and XDOF is reportedly raising at a $1.2B valuation. Raw egocentric video costs roughly $5–$40/hr to produce and teleop roughly $28–$200/hr, while undifferentiated footage sells for as little as $2–$5/hr. Collector pay in India can be $1–$4/hr.

### Cited Findings
**Funding**
- Mecka AI: $60M Series B led by Sequoia (Oct 7 2026), with Nvidia and M12. It had reportedly been nearing a $500M valuation (Sep 2026). — [TechCrunch](https://techcrunch.com/2026/10/07/robot-data-startup-mecka-ai-nabs-60m-from-sequoia/)
- Mecka AI raised $60M across two rounds in June 2026 and claims $100M run-rate revenue (June), on track for $300M by year-end. These are company claims reported in a search summary; the TechCrunch article itself gave no revenue. — [Dealroom/Automate roundup](https://www.automate.org/robotics/industry-insights/robotics-raises-icon-mecka-ai-multiply-labs-mmi-ultra/aph); an earlier $8M round is covered by [Upstarts Media](https://www.upstartsmedia.com/p/mecka-ai-robotics-data-startup)
- XDOF: $70M Series A (June 2026) from Thrive, a16z, Lux and Spark. It is reportedly in Series B talks at ~$1.2B led by 8VC (TechCrunch, Sep 4 2026; not finalized). Annualized revenue is reportedly approaching $50M across ~20 clients, including frontier labs. — [Dealroom](https://dealroom.co/news/149061-xdof-nears-1-2b-series-b-just-months-after-70m-series-a/); [KuCoin summary](https://www.kucoin.com/news/flash/xdof-in-b-round-talks-at-1-2b-valuation-months-after-70m-a-round)
- Human Archive: $8.2M (May 2026) from Wing VC, NVP and YC, plus angels from OpenAI, Nvidia, Google and Meta. — [TechCrunch](https://techcrunch.com/2026/05/26/human-archive-taps-into-indias-services-startups-to-collect-data-for-physical-ai/)
- Figure: $1B+ committed to data and compute over 12 months, and $15M already paid to contributors (Aug 2026). — [Humanoids Daily](https://www.humanoidsdaily.com/news/figure-ai-unveils-index-crowdsourcing-real-world-human-video-to-train-helix)
- Sector: robotics startups raised ~$14B in 2025 (a record) and $18.8B in H1 2026 (Crunchbase). — [Manufacturing Dive](https://www.manufacturingdive.com/news/humanoid-hype-surging-investor-capital-state-of-industrial-robots/824040/). Another outlet says "nearly $48B" for robotics plus physical AI in 2026 YTD, which is unsourced and uses a different definition. — [Crypto Briefing](https://cryptobriefing.com/robotics-labs-motion-video-data-ai/)

**Pricing (vendor-reported, mostly raw hours)**
- DexSet (Jul 2026) prices per raw hour as follows. Teleop costs $28–$60/hr (rig $4–9, operator $18–38, QA $6–12). Egocentric video costs $15–$40/hr (hardware $1–4, collector $10–26, QA $5–10). Multi-view exocentric (3–8 cameras) costs $20–$50/hr, and stereo adds 15–25%. Annotation is extra: language labels $8–12, subtask segmentation $10–16, dense masks or contact $18–25 per data-hour. — [DexSet](https://dexset.ai/blogs/robot-training-data-costs-pricing-complete-2026/)
- Flikforge gives different figures. Egocentric video costs $5–$20/hr to create. Bulk undifferentiated footage trades at $2–$5/hr. Teleop costs $50–$200/hr, with only a few usable hours per operator per day. Its claim that curated data is worth "~100x" bulk is unsourced. — [Flikforge](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/)
- Another vendor quotes calibrated wearable programs at $25–40/hr, rising toward $60/hr with dense labels. Packaged Indian egocentric data reportedly resells at $15–50/hr (unverified). — [search summary of Unidata / DataXPower / Exylos](https://www.exylos.ai/blog/robotics-dataset-pricing/)
- Collector pay: Human Archive pays a $1/hr base. Competitors reportedly pay ₹250–400/hr (~$2.6–$4.2) per Economic Times. — [TechCrunch](https://techcrunch.com/2026/05/26/human-archive-taps-into-indias-services-startups-to-collect-data-for-physical-ai/)

### Inferences
- The gap between production cost ($5–40/hr) and the valuations implied by revenue ($50M–$300M run-rates) suggests buyers pay mainly for scale, diversity, sync and labels, not raw footage.
- Prices are falling as supply floods in. Value is moving to curation, pose/trajectory extraction, and format conversion (LeRobot/RLDS).

### Gaps
- No public list prices from Scale AI, Micro1, Mecka or XDOF.
- Figure's per-video bounty rates are not disclosed.
- Mecka's revenue claims are not confirmed in a primary source I could read.

## 3. Is learning from human video considered a key bottleneck? Quotes

### Takeaway
Yes. The "robot data gap" is the consensus framing. Human video is widely seen as the most scalable way to close it, but people disagree on whether raw video is useful without curation and robot co-training.

### Cited Findings
- Ken Goldberg (UC Berkeley): "We have collected approximately a year of robot data," versus about "100,000 years" for text and images. He calls this the "Robot Data Gap" and proposes simulation, video, teleop and real production data as ways to close it. — [diginomica](https://diginomica.com/robot-futures-1-why-your-dirty-socks-are-preventing-future); [NYU talk abstract](https://robotics.engineering.nyu.edu/event/ken-goldberg-talk/)
- Goldberg's Aug 27 2026 Science Robotics papers argue capable humanoids are further off than optimistic timelines suggest. — [Interesting Engineering](https://interestingengineering.com/innovation/humanoid-reality-check-elon-musk)
- DeepReach: physical AI "is limited by diverse, real-world data at massive scale showing how humans actually do work in the physical world." — [YC](https://www.ycombinator.com/companies/industry/robotics)
- Wing VC's Zach DeWitt on Human Archive: "No one else in the world has been able to synchronize and collect headset RGB-D, force feedback…" — [TechCrunch](https://techcrunch.com/2026/05/26/human-archive-taps-into-indias-services-startups-to-collect-data-for-physical-ai/)
- Brett Adcock (paraphrased): hardware is no longer the main hurdle; the bottleneck is "onboard intelligence and general-purpose reasoning," and Index feeds Helix. — [Humanoids Daily](https://www.humanoidsdaily.com/news/figure-ai-unveils-index-crowdsourcing-real-world-human-video-to-train-helix)
- Evidence that human video helps:
  - EgoMimic (Georgia Tech): co-training on human egocentric plus robot data beat robot-only by 34–228%.
  - 2 hours of robot data plus 1 hour of human-hand data beat 3 hours of robot data alone.
  - Scaling-law papers find environment and object diversity matters more than demo count. About 32 environments × 50 demos reached 90% success in new environments (arXiv 2410.18647).
  - Source: [Flikforge summary](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/)
- A HumanScale paper (arXiv 2606.20521) claims egocentric human video can outperform real-robot data for embodied pretraining. — [arXiv](https://arxiv.org/pdf/2606.20521)
- Counterpoint on quality: buyers reportedly discard about 90% of bulk footage for a given task. One customer said: "In any given hour of footage, we are lucky to find 2 or 3 segments that are suitable." Consumer hardware (iPhones, GoPros) shows jittery sensor sync and pose drift. NVIDIA's Max Zhaoshuo Li predicts egocentric data will be priced by how far it expands a "diversity-fidelity frontier," not by the hour. — [Flikforge (vendor-authored)](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/)

### Inferences
- "More hours" is no longer the scarce input. Diverse, well-synced, accurately pose-labeled, task-segmented clips are. That favors a tool that scores and filters quality over one that only captures.

### Gaps
- I did not find primary 2026 quotes from Sergey Levine / Karol Hausman (PI), Jensen Huang, or Deepak Pathak (Skild) on human video in this pass.

## 4. Underserved niches a scrappy webcam-based tool could fill

### Takeaway
The leaders cover scale capture with phones, caps and rigs. The open spots are (a) QA and filtering of footage people already have, (b) accurate 3D hand/body joint extraction and retargeting from monocular video, (c) sync and calibration for cheap multi-camera setups, (d) underserved verticals with fixed workstations, where a static webcam works well, and (e) compliance and provenance tooling.

### Cited Findings
- Quality and curation is a stated pain point. Buyers discard ~90% of bulk footage, consumer devices drift, and curated data is said to command a premium. — [Flikforge](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/)
- Annotation (segmentation $10–16/hr, dense contact $18–25/hr) costs about as much as capture, so automating it has direct value. — [DexSet](https://dexset.ai/blogs/robot-training-data-costs-pricing-complete-2026/)
- Exocentric multi-view capture (3–8 cameras) is already a priced product category at $20–$50/hr, which suggests fixed, webcam-style setups have a market. — [DexSet](https://dexset.ai/blogs/robot-training-data-costs-pricing-complete-2026/)
- Agile Robots turning ordinary worker video into trajectories shows demand for video-to-trajectory software that needs no rig. — [Crypto Briefing](https://cryptobriefing.com/robotics-labs-motion-video-data-ai/)
- Factory and assembly egocentric capture is under way (Encord in factories, Egolab with factory head cams), but dedicated healthcare and assisted-living data looks thin. There is academic work on robot-assisted bathing demonstrations. — [TechCrunch](https://techcrunch.com/2026/07/26/are-brain-waves-the-next-unlock-for-physical-ai/); [arXiv 2608.09127](https://arxiv.org/pdf/2608.09127)
- Consumer crowdsourcing is proven at scale by Figure Index (44K WAU) and is being pursued by Human Archive's open earn platform. — [Humanoids Daily](https://www.humanoidsdaily.com/news/figure-ai-unveils-index-crowdsourcing-real-world-human-video-to-train-helix); [TechCrunch](https://techcrunch.com/2026/05/26/human-archive-taps-into-indias-services-startups-to-collect-data-for-physical-ai/)
- Regulatory and consent risk: India's MeitY is examining consent practices at egocentric-data startups, and Urban Company refused to take part. — [TechCrunch](https://techcrunch.com/2026/05/26/human-archive-taps-into-indias-services-startups-to-collect-data-for-physical-ai/)
- Standard formats (LeRobot, RLDS, MCAP) and provenance metadata are selling points. — [Flikforge](https://flikforge.com/robot-data-glut-egocentric-video-physical-ai/)
- Crowdsourced smartphone teleop (COBALT, arXiv 2605.19138) shows academic interest in phone-based collection. — [arXiv](https://arxiv.org/pdf/2605.19138)
- The best open hand-pose dataset (EgoDex) is non-commercial (CC-BY-NC-ND), which leaves room for commercially licensed hand-pose data. — [GitHub](https://github.com/apple/ml-egodex)

### Inferences
- Startup-worthy wedges (inference):
  1. A "data QA linter" that scores clips for occlusion, hand visibility, pose-tracking confidence, motion blur and task segmentation, and exports LeRobot/RLDS.
  2. Monocular webcam to 3D hand/arm trajectory, with retargeting to common grippers and hands.
  3. Fixed-station capture kits for SMB manufacturing or kitting lines, sold as "record your best worker's process."
  4. Privacy tooling: face blur, consent logging, provenance.
- Avoid competing with Figure, Scale or Mecka on raw collection volume.

### Gaps
- No market-size data for robot-data QA tooling specifically.
- No evidence found on warehouse-specific or healthcare-specific human-demo data vendors.

## 5. Adjacent non-robotics markets (fallback angles)

### Takeaway
The same monocular pose tech already supports funded but modestly sized businesses in animation mocap, workplace ergonomics and safety, and clinical rehab. These are viable fallbacks, though recent (2025–2026) funding data for these players was sparse in my searches.

### Cited Findings
- Move AI (phone and multi-camera markerless mocap): $10M seed (Oct 2023) from Play Ventures, Warner Music Group, RKKVC, Level2 and Animoca. Dealroom lists $17.4M total across 7 rounds. Revenue comes from subscriptions and licensing. — [VentureBeat](https://venturebeat.com/ai/3d-motion-capture-app-move-ai-raises-10m-in-seed-funding); [Dealroom](https://dealroom.co/companies/move-ai/)
- Market size for markerless mocap is about $87.6M (2025), $102.9M (2026) and $370.9M (2034, projected). This is an analyst-firm estimate, so treat it as rough. — [Fortune Business Insights](https://www.fortunebusinessinsights.com/markerless-motion-capture-market-117614)
- MOVIN (real-time markerless mocap) raised an undisclosed seed from NAVER D2SF (Dec 2023). — [MOVIN](https://www.movin3d.com/discover/real-time-markerless-motion-capture-startup-movin-secures-seed-funding-from-naver-d2sf-bluepoint-partners-and-krew-capital)
- Ergonomics:
  - TuMeke (phone-app CV injury-risk assessment for manufacturing): $10M Series A led by Intel Capital (Dec 2023) and a $2.5M seed. — [Intel Capital](https://www.intelcapital.com/tumeke-raises-10m-in-series-a-funding-led-by-intel-capital/)
  - Voxel (hazard detection on existing security cameras): $12M round led by Rite-Hite, $30M total (2023). — [IoT World Today](https://www.iotworldtoday.com/iiot/computer-vision-startup-raises-12m-for-workplace-safety)
  - Inseer (CV ergonomics): $420K (Jul 2025), from a weak directory source. — [vcbacked](https://www.vcbacked.co/company/inseer)
- Clinical: a 2026 PRISMA scoping review covers video-based markerless capture for rehab biomechanics, including mobile-device lifting kinematics. A 2026 Balgrist trial is testing markerless capture in foot and ankle surgery. — [arXiv 2609.18667](https://arxiv.org/pdf/2609.18667); [ClinicalTrials.gov](https://clinicaltrials.gov/study/NCT07848282)

### Inferences
- Ergonomics and safety (TuMeke-style phone or webcam assessment) is the closest fallback for a webcam pose tool, because the same "record a worker on an assembly line" footage can serve both an ergonomics report and a robot-demo dataset. That is a possible dual-revenue wedge.
- Animation mocap is crowded (Move AI, Rokoko, DeepMotion) and the market is small (~$100M).

### Gaps
- No 2025–2026 funding or pricing found for Rokoko, DeepMotion, Kinetix, Sword Health, Hinge Health's motion tech, or Kemtai in this pass.
- Move One's current pricing is not public in the results.
