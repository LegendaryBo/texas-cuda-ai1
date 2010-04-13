package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.fitnessdist;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveFunctionDistributionGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveFunctionValueDistributionStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveValueStatistic;
import pl.wroc.uni.ii.evolution.testhelper.EvGoalFunction;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;
import junit.framework.TestCase;

public class EvFitnessDistributionGathererTest extends TestCase {

  public void testitWorks() {
    
    EvPopulation<EvStunt> pop = new EvPopulation<EvStunt>();

    pop.add(new EvStunt(2.5));
    pop.add(new EvStunt(2.5));
    pop.add(new EvStunt(0.0));
    pop.add(new EvStunt(0.5));
    pop.add(new EvStunt(0.0));
    pop.add(new EvStunt(0.1));

    EvGoalFunction fun = new EvGoalFunction();
    pop.setObjectiveFunction(fun);
    
    EvObjectiveFunctionDistributionGatherer<EvStunt> operator_stats = new EvObjectiveFunctionDistributionGatherer<EvStunt>(null);
    EvObjectiveFunctionValueDistributionStatistic stats = (EvObjectiveFunctionValueDistributionStatistic) operator_stats.generate(pop);
    
    List<EvObjectiveValueStatistic> value_stats = stats.getStatistics();
    
    assertTrue("The Statistics doesn't discribe fintess values in the population", value_stats.size() == 4);
        
    assertEquals(2, value_stats.get(0).getNumber());
    assertEquals(0.0, value_stats.get(0).getFitness());
    
    
    assertEquals(1, value_stats.get(1).getNumber());
    assertEquals(0.1, value_stats.get(1).getFitness());
    
      
    assertEquals(1, value_stats.get(2).getNumber());
    assertEquals(0.5, value_stats.get(2).getFitness());
    
    
    assertEquals(2, value_stats.get(3).getNumber());
    assertEquals(2.5, value_stats.get(3).getFitness());
    
    
  }
 
  
  
  
}
