# Advisor Feedback on FreethrowEEG Study Plan

## Kyle Mathewson, PhD

### Mobile Brain-Body Imaging Lab

### University of Alberta

Great initial planning, Sarah and Alex. I can see you've thought through many of the key experimental design elements. Having worked extensively with mobile EEG in real-world settings, I have some suggestions to strengthen your plan and reduce potential risks:

### Critical Considerations

#### 1. Data Quality Validation

Before full data collection:

- Run a full pilot week (5 sessions) with your chosen participant
- Develop automated artifact detection scripts
- Set clear criteria for trial rejection
- Test all your analysis pipelines on the pilot data
- Establish baseline noise levels in the basketball court environment

#### 2. Technical Setup Improvements

- Add a second computer for redundant data storage
- Create automated backup after every 25-shot block
- Use UPS (Uninterruptible Power Supply) for all recording equipment
- Test Muse connection stability in the gym environment (Bluetooth interference?)
- Consider using LSL (Lab Streaming Layer) for precise timing synchronization

#### 3. Experimental Design Refinements

Your basic design is solid, but consider these additions:

- Add 10 practice trials at the start of each session (not analyzed)
- Include regular impedance photos (every 25 trials)
- Record room temperature and humidity (affects sweat/impedance)
- KYLE REAL \*\* Add brief rest EEG (2 min) at start/end of each session
- Consider shorter blocks (20 shots instead of 25) to reduce fatigue

#### 4. Data Management Protocol

Implement this hierarchy immediately:

```
project_root/
├── raw_data/
│   ├── sub001/
│   │   ├── session_01/
│   │   │   ├── eeg/
│   │   │   ├── behavioral/
│   │   │   ├── video/
│   │   │   └── metadata.json
│   │   └── ...
├── processed_data/
├── analysis_scripts/
├── quality_checks/
└── backups/
```

#### 5. Pre-study Validation Steps

Before starting the 4-week protocol:

1. Run full technical validation:

   - Test all equipment for 2 full sessions
   - Verify data saving/backup procedures
   - Check video synchronization
   - Validate trigger timing precision

2. Develop and test quality control scripts:

   - Real-time impedance monitoring
   - Automated artifact detection
   - Data completeness checks
   - Synchronization verification

3. Create automated reports for:
   - Signal quality metrics
   - Behavioral performance
   - System performance (e.g., trigger timing, data saving)

#### 6. Risk Mitigation Strategies

- Have backup Muse device ready
- Create detailed troubleshooting guide
- Establish clear criteria for session restart/continuation
- Define minimum data quality thresholds
- Plan for possible participant absence/rescheduling

### Specific Recommendations

1. **Data Collection**

- Add event markers for:
  - Start/end of baseline period
  - Shot initiation
  - Ball release
  - Basket/miss
  - Any unusual events (noise, interruptions)
- Save raw EEG data at maximum sampling rate
- Include accelerometer data from Muse

2. **Analysis Pipeline**

- Develop these scripts before starting data collection:
  - Automated preprocessing pipeline
  - Quality metric calculation
  - Trial segmentation
  - Basic visualization tools
- Test pipeline with simulated "worst-case" data

3. **Documentation**

- Create detailed SOPs for:
  - Equipment setup
  - Participant preparation
  - Session execution
  - Troubleshooting
  - Data backup procedures

### Timeline Adjustment

Week -2:

- Setup and test recording environment
- Develop/test analysis scripts
- Create SOPs
- Technical validation

Week -1:

- Pilot testing (5 sessions)
- Analysis pipeline validation
- Refinement of procedures

Weeks 1-4:

- Main data collection
- Daily quality checks
- Weekly preliminary analysis

### Additional Resources Needed

- Backup Muse device
- UPS system
- Secondary recording computer
- Environmental sensors (temp/humidity)
- External hard drives for backup

### Success Criteria

Define clear criteria for:

- Minimum number of valid trials per session
- Acceptable impedance ranges
- Maximum movement artifact percentage
- Required signal-to-noise ratio

### Next Steps

1. Implement data management structure
2. Develop quality control scripts
3. Run technical validation tests
4. Create detailed SOPs
5. Conduct pilot week
6. Review pilot data
7. Begin full study

Remember: In mobile EEG research, meticulous preparation and robust data quality checks are crucial. It's better to spend extra time in preparation than to discover issues after weeks of data collection.

Let's meet again after you've implemented these suggestions and before starting the pilot week. I'd like to review your quality control scripts and SOPs in detail.

Best,
Kyle
