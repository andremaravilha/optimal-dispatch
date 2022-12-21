library("irace")

# Load tuning scenario
scenario <- readScenario(filename = "scenario.txt", scenario = defaultScenario())
checkIraceScenario(scenario = scenario)

# Start tuning
#irace.main(scenario = scenario)

