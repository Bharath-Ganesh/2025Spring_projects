## Title: Autonomous Vehicles Simulation

## Team Member(s): Bharath Ganesh, Harisankar Kartha, Shantanu Roy

## Monte Carlo Simulation Scenario & Purpose:
We simulate object detection performance of different sensor configurations (camera-only, multi-sensor fusion, and hybrid setups) on an autonomous vehicle under clear and foggy conditions. The goal is to evaluate safety and cost-effectiveness by measuring detection rates across 5,000 iterations per configuration.

## Hypothesis before running the simulation:

1. A multi-sensor configuration will maintain higher detection rates in adverse weather compared to a camera-only system.
2. A hybrid configuration can achieve detection rates within 2% of the best-performing system while using fewer sensors.

## Simulation's variables of uncertainty:

Object Positions: Uniform distribution over the vehicle's detection field; range chosen to cover all lanes and adjacent sidewalks.
Object Types: Weighted categorical distribution (e.g., pedestrians, vehicles, cyclists) based on real-world traffic mix data.
Sensor Noise: Gaussian noise added to each measurement; standard deviation selected from sensor specifications to reflect typical measurement error.
Sensor Dropout: Bernoulli trial for each sensor per iteration; dropout probability set to mimic temporary failures or occlusions.
Each variable uses pseudo-random number generation to model real-world uncertainty. The selected ranges and distributions are grounded in standard sensor specifications and traffic statistics to ensure plausible scenarios.

## Results & Discussion:

Detection Rates Summary:
Clear weather: Camera Only achieved 97.8% detection; Sensor Fusion achieved 99.6% detection.
Comparison Across Configurations:

## Sensor Fusion (13 sensors): 99.08% average detection rate.

Best Hybrid (7 cameras, 3 radars, 1 lidar; 11 sensors): 98.15% average detection rate.
Camera-Only (8 sensors): 94.49% average detection rate.

## Statistical Confidence:
Convergence observed after ~1,000 iterations; results remained stable and consistent across multiple runs, validating simulation reliability.

## Insights & Limitations:

Multi-sensor fusion maintains high detection rates even in fog (only a 2.63% drop), whereas camera-only systems suffer significant degradation (10.16% drop)
Objects are modeled as dimensionless points; occlusion, dynamic scenarios, and additional weather effects (rain, snow) are not yet incorporated.
Detection probability models are simplified (binary camera detection, Beer–Lambert law only for fog)

## Sources Used:
1. Hasirlioglu, S., & Riener, A. (2020). "Introduction to Rain and Fog Attenuation on Automotive Surround Sensors." IEEE Intelligent Transportation Systems Magazine, 12(4), 6-22.
2. Bijelic, M., Gruber, T., & Ritter, W. (2018). "Benchmarking Image Processing Algorithms for Adverse Weather Conditions." IEEE Transactions on Intelligent Transportation Systems, 19(12), 3867-3881.
3. Brooker, G. (2007). "Understanding Millimetre Wave FMCW Radars." International Conference on Sensing Technology.
4. Gultepe, I., et al. (2007). "Fog Research: A Review of Past Achievements and Future Perspectives." Pure and Applied Geophysics, 164(6-7), 1121-1159.

