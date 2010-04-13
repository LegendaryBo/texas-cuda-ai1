package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;
import pl.wroc.uni.ii.evolution.testhelper.EvGoalFunction;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;
import junit.framework.TestCase;

public class EvMaxAvgMinGathererTest extends TestCase {

  public void testPopulation3() {

    EvPopulation<EvStunt> pop = new EvPopulation<EvStunt>();

    pop.add(new EvStunt(2.5));
    pop.add(new EvStunt(0.0));
    pop.add(new EvStunt(0.5));

    EvGoalFunction fun = new EvGoalFunction();
    pop.setObjectiveFunction(fun);

    EvObjectiveFunctionValueMaxAvgMinGatherer<EvStunt> operator_stat = new EvObjectiveFunctionValueMaxAvgMinGatherer<EvStunt>(null);

    EvObjectiveFunctionValueMaxAvgMinStatistic stats = (EvObjectiveFunctionValueMaxAvgMinStatistic) operator_stat.generate(pop);

    assertEquals(2.5 , stats.getMax());
    assertEquals(0.0, stats.getMin());
    assertEquals(1.0, stats.getAvg());



  }

  public void testPopulationOne() {

    EvPopulation<EvStunt> pop = new EvPopulation<EvStunt>();

    pop.add(new EvStunt(2.5));

    EvGoalFunction fun = new EvGoalFunction();
    pop.setObjectiveFunction(fun);

    EvObjectiveFunctionValueMaxAvgMinGatherer<EvStunt> operator_stat = new EvObjectiveFunctionValueMaxAvgMinGatherer<EvStunt>(null);

    EvObjectiveFunctionValueMaxAvgMinStatistic stats = (EvObjectiveFunctionValueMaxAvgMinStatistic) operator_stat.generate(pop);

    assertEquals(2.5 , stats.getMax());
    assertEquals(2.5, stats.getMin());
    assertEquals(2.5, stats.getAvg());



  }
  
}
