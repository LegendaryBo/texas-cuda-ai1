package pl.wroc.uni.ii.evolution.engine.operators.general.composition;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvApplyOnSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.misc.EvIdentity;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import junit.framework.TestCase;

public class EvApplyOnSelectionCompositionTest extends TestCase {


  private class EvSimpleMutation extends EvMutation<EvBinaryVectorIndividual> {

    @Override
    @SuppressWarnings("unused")
    public EvBinaryVectorIndividual mutate(EvBinaryVectorIndividual individual) {
     for (int i = 0; i < individual.getDimension(); i++ ) {
       individual.setGene(i, 0);
     }
     return individual;
    }
    
  }
  

  public void testItWorks() {
    

    EvBinaryVectorIndividual ind1 = new EvBinaryVectorIndividual(new int[] {1, 1, 1, 1});
    EvBinaryVectorIndividual ind2 = new EvBinaryVectorIndividual(new int[] {1, 1, 0, 1});
    EvBinaryVectorIndividual ind3 = new EvBinaryVectorIndividual(new int[] {1, 1, 0, 0});
    EvBinaryVectorIndividual ind4 = new EvBinaryVectorIndividual(new int[] {0, 0, 0, 0});
    EvBinaryVectorIndividual ind5 = new EvBinaryVectorIndividual(new int[] {0, 0, 1, 0});

    EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>();
    population.add(ind1);
    population.add(ind2);
    population.add(ind3);
    population.add(ind4);
    population.add(ind5);
    population.setObjectiveFunction(new EvOneMax());



    EvApplyOnSelectionComposition<EvBinaryVectorIndividual> operator = 
      new EvApplyOnSelectionComposition<EvBinaryVectorIndividual>(new EvKBestSelection<EvBinaryVectorIndividual>(2), new EvIdentity<EvBinaryVectorIndividual>(), new EvSimpleMutation());


    EvPopulation<EvBinaryVectorIndividual> result = operator.apply(population);
    assertEquals(5, result.size());
    
    for (EvBinaryVectorIndividual ind: result) {
      System.out.println(ind);
    }
  }
  
  
}
