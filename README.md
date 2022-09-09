# Exploring Out-Of-Distribution in Image Classification for Neural Networks via Concepts
Website accompanying the paper with the above title:

**Abstract** *The currently dominating artificial intelligence and machine learning technology, neural networks, builds on inductive statistical learning processes. Being void of knowledge that can be used deductively these systems cannot distinguish exemplars part of the target domain from those not part of it. This ability is critical when the aim is to build human trust in real-world settings and essential to avoid usage in domains wherein a system cannot be trusted. In the work presented here we conduct two qualitative contextual user studies and one controlled experiment to uncover research paths and design openings for the sought distinction. Through our experiments, we find a need to refocus from average case metrics and benchmarking datasets toward systems that can be falsified. The work uncovers and lays bare the need to incorporate and internalise a domain ontology in the systems and/or present evidence for a decision in a fashion that allows a human to use our unique knowledge and reasoning capability.*

###### This site adds code and additional material for reproducibility and additional material.

**Study 1:** This study investigated whether it is useful, for a human, to categorise visible sub-concepts in images picturing animals as necessary, sufficient and, additionally, the usefulness of the notion of spurious correlations concerning concepts not deemed as necessary or sufficient. By using these notions we for this study hypotesize that they can make it easier to identify o.o.d exemplars. Concepts discussed here were then sub-concepts to animal classifications, as, for example, 'watering hole', 'beak' or 'feather'.
[Link to the study](https://htmlpreview.github.io/?https://github.com/k3larra/ood/blob/main/animals_version01.html)
Link to code for the predictions:

**Study 2:** This study focused on a more narrow domain headgear and the seven directly related classes in ImageNet-1K: sombrero', cowboy hat, 'bathing cap', 'crash helmet, 'bonnet', 'shower cap' and 'football helmet. In this study we did not use any concept theory and instead relied on the participants intuitive understanding of headgear related concepts. This closer adhere prototype theories and concepts as central features of the phenomena in question~\cite{Murphy2018}.
[Link to the study](https://htmlpreview.github.io/?https://github.com/k3larra/ood/blob/main/headgear_version01.html)

Link to code for study 1 and study 2

**Study 3:** In this part we make a comparative analyse of predictions and explanations related an image picturing a horse that were used in the first study. Using our eight pre-trained models and the model independent XAI-method Occlusion we can, using concepts, compare and discuss predictions from a o.o.d perspective. According to the dictionary [Merriam-Webster](https://www.merriam-webster.com/dictionary/sorrel) sorrel has two definitions either it is a light bright chestnut coloured horse or a plant with sour juice, typically common sorrel (Rumex acetosa). These two different concepts, a type of horse and a group of plants carries a wealth of knowledge that, we as humans, connect to our real world knowledge. For example, if a person is knowledgeable about horses, the connection between this horse colour and the horse-breed quarter-horse and that that this type of horse in Europe commonly is denoted chestnut". And, of course, the cultural global-north discourse connected to these labels and concepts can be taken into account depending of whom the classification should be useful for. Type of plants denoted as sorrels similarly follows a wealth of causal and descriptive factors that also can be contextualised. The WordNet semantic relations to sorrel brings up overlapping nouns as Merriam-Webster and additionally adds a definition of sorrel as an adjective for a brownish-orange colour. In this work we lift out "sorrel" as an example of a concepts that are incomplete, contextual and contains, both causal and descriptive factors.

Click the images for to get the ML-model comparisons.

[![](testset/thumbnails/0.jpg)](https://k3larra.github.io/ood/sorrel_version01.html?study_nbr=0)
[![](testset/thumbnails/1.jpg)](https://k3larra.github.io/ood/sorrel_version01.html?study_nbr=1)
[![](testset/thumbnails/2.jpg)](https://k3larra.github.io/ood/sorrel_version01.html?study_nbr=2)
[![](testset/thumbnails/3.jpg)](https://k3larra.github.io/ood/sorrel_version01.html?study_nbr=3)
[![](testset/thumbnails/4.jpg)](https://k3larra.github.io/ood/sorrel_version01.html?study_nbr=4)
[![](testset/thumbnails/5.jpg)](https://k3larra.github.io/ood/sorrel_version01.html?study_nbr=5)


[Information on pretrained models used in Study 3](https://github.com/k3larra/ood/blob/main/models.md)
[Link to code for study 3]
