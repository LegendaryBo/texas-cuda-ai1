package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * A statistic about every fitness value in population.
 * 
 * @author Marcin Golebiowski
 */
public class EvObjectiveFunctionValueDistributionStatistic extends EvStatistic {

  private static final long serialVersionUID = -5003154093318013779L;

  private List<EvObjectiveValueStatistic> values;


  public EvObjectiveFunctionValueDistributionStatistic(int iteration) {
    this.setIteration(iteration);
    this.values = new ArrayList<EvObjectiveValueStatistic>();
  }


  public void addFinesssValueStatistic(EvObjectiveValueStatistic value_stat) {
    values.add(value_stat);
  }


  public List<EvObjectiveValueStatistic> getStatistics() {
    return values;
  }

}
