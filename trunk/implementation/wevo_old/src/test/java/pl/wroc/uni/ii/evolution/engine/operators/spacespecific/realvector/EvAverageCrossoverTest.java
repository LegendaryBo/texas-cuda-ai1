package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;


import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorAverageCrossover;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSimpleApplyStrategy;

import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;

/**
 * @author Kamil Dworakowski, Jarek Fuks
 * 
 */
public class EvAverageCrossoverTest extends TestCase {

  public void testCrossover() throws Exception {

    EvRealVectorIndividual indiv1 = new EvRealVectorIndividual(new double[] { 1d });
    indiv1.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());
    EvRealVectorIndividual indiv2 = new EvRealVectorIndividual(new double[] { 2d });
    indiv2.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());

    EvRealVectorAverageCrossover operator = new EvRealVectorAverageCrossover(2);
    operator.setCrossoverStrategy(new EvSimpleApplyStrategy());
    
    EvPopulation<EvRealVectorIndividual> parents = new EvPopulation<EvRealVectorIndividual>();
    parents.add(indiv1);
    parents.add(indiv2);

    EvRealVectorIndividual baby = (EvRealVectorIndividual) operator
        .apply(parents).get(0);

    assertEquals(1.5d, baby.getValue(0), 0.000000001d);

    assertEquals(-0.5d, baby.getObjectiveFunctionValue(), 0.000000000001);
  }

}
