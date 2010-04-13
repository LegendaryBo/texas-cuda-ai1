package pl.wroc.uni.ii.evolution.engine.individuals;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyMaxSum;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyObjectiveFunction;
import pl.wroc.uni.ii.evolution.objectivefunctions.naturalnumbervector.EvNaturalPattern;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
/**
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvCloningTest extends TestCase {
  
  public void testCloningBinaryIndividual() {
    EvBinaryVectorIndividual b1 = new EvBinaryVectorIndividual(new int[] {1, 0});
    b1.setObjectiveFunction(new EvOneMax());
    EvBinaryVectorIndividual b2 = b1.clone();

    assertEquals(1d,b2.getObjectiveFunctionValue(),0.01);
    b2.setGene(0, 0);
    assertTrue(b1.getGene(0) == 1);
    assertEquals(1d,b1.getObjectiveFunctionValue(),0.01);
    EvBinaryVectorIndividual b3 = b1.clone();
    assertEquals(1d,b3.getObjectiveFunctionValue(),0.01);
  }
  
  public void testCloningSimplifiedMessyIndividual() {
    EvSimplifiedMessyIndividual individual = new EvSimplifiedMessyIndividual(3);
    individual.setObjectiveFunction(new EvSimplifiedMessyObjectiveFunction(2, new EvSimplifiedMessyMaxSum(),10));
    individual.addGeneValue(0, 1);
    individual.addGeneValue(1, 0);
    EvSimplifiedMessyIndividual clone = individual.clone();
    assertEquals(clone,individual);
    assertEquals(clone.getObjectiveFunctionValue(), individual.getObjectiveFunctionValue());
    clone.addGeneValue(2, 1);
    assertTrue(individual.getGeneValues(2).size() == 0);
    assertEquals(1, clone.getGeneValue(2));
    EvSimplifiedMessyIndividual clone2 = individual.clone();
    assertEquals(clone2.getObjectiveFunctionValue(), individual.getObjectiveFunctionValue());
  }
  
  public void testCloningMiLambdaRoKappaIndividual() {
    EvMiLambdaRoKappaIndividual individual;
    individual = new EvMiLambdaRoKappaIndividual(1);
    individual.setAlpha( 0, .1 );
    individual.setValue( 0, .1 );
    individual.setProbability( 0, .1 );
    
    individual.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    EvMiLambdaRoKappaIndividual new_individual = individual.clone();

    assertTrue( new_individual.getAlpha( 0 ) == .1 );
    assertEquals(individual.getObjectiveFunctionValue(), new_individual.getObjectiveFunctionValue());
    new_individual.setAlpha( 0, .2 );
    
    assertTrue( individual.getAlpha( 0 ) == .1 );
    
    EvMiLambdaRoKappaIndividual new_individual2 = individual.clone();
    assertEquals(individual.getObjectiveFunctionValue(), new_individual2.getObjectiveFunctionValue());
    
  }
  
  public void testCloningNaturalNumberIndividual() {
    EvNaturalNumberVectorIndividual n1 = new EvNaturalNumberVectorIndividual(2);
    n1.setNumberAtPosition(0, 123);
    n1.setNumberAtPosition(1, 120);
    n1.setObjectiveFunction(new EvNaturalPattern(new int[] {123, 120}));
    EvNaturalNumberVectorIndividual n2 = n1.clone();
    assertEquals(n1.getObjectiveFunctionValue(), n2.getObjectiveFunctionValue());
    n2.setNumberAtPosition(1, 200);
    assertFalse(n2.getNumberAtPosition(1) == n1.getNumberAtPosition(1));
    EvNaturalNumberVectorIndividual n3 = n1.clone();
    assertEquals(n1.getObjectiveFunctionValue(), n3.getObjectiveFunctionValue());
  }
  
  public void testCloningRealVectorIndividual() {
    
    EvRealVectorIndividual r1 = new EvRealVectorIndividual(new double[] {1.0,0.0});
    r1.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());
    EvRealVectorIndividual r2 = r1.clone();
    assertEquals(-1.0, r2.getObjectiveFunctionValue() );
    r2.setValue(1, 3.0);
    assertFalse(r1.getValue(1) == 3.0);
    assertEquals(-1.0, r1.getObjectiveFunctionValue() );
    EvRealVectorIndividual r3 = r1.clone();
    assertEquals(-1.0, r3.getObjectiveFunctionValue() );
  }
  
  public void testCloningRealVectorWithProbabilitiesIndividual() {
    EvRealVectorWithProbabilitiesIndividual individual;
    individual = new EvRealVectorWithProbabilitiesIndividual(1);
    individual.setValue( 0, .1 );
    individual.setProbability( 0, .1 );
    
    individual.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    EvRealVectorWithProbabilitiesIndividual new_individual = individual.clone();

    assertTrue( new_individual.getProbability( 0 ) == .1 );
    assertEquals(individual.getObjectiveFunctionValue(), new_individual.getObjectiveFunctionValue());
    new_individual.setProbability( 0, .2 );
    
    assertTrue( individual.getProbability( 0 ) == .1 );
    
    EvRealVectorWithProbabilitiesIndividual new_individual2 = individual.clone();
    assertEquals(individual.getObjectiveFunctionValue(), new_individual2.getObjectiveFunctionValue());
    
  }
}
