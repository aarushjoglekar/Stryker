# Stryker

In fast-paced soccer matches, understanding what’s happening on the field isn’t just about seeing the present - it’s about recognizing the flow of play. Stryker takes a sequence-based approach to action recognition. It begins by converting labeled SoccerNet video clips into 60-second clips, and extracting visual features from each clip using DINOv2 - the video transformer. These features are then passed into another 6-layer Transformer, which learns how actions evolve over time. A final sigmoid layer predicts the probability of each  action, and classifies actions with atleast 50% probability as happening during the clip. This temporal and context-aware system aims to deepen our understanding of complex gameplay dynamics and support real-time decision-making in sports analytics.

Presentation: https://www.canva.com/design/DAGulBKiNAE/NsMBLvPV5G4cLZoYoXpKHw/view

By: Aarush, Aaditya, Benjamin, Varshith
