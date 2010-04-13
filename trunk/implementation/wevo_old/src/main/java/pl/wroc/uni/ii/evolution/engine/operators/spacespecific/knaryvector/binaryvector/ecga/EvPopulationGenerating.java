package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

import java.util.ArrayList;
import java.util.Collections;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Generate new population using ECGAStructure
 * 
 * @author Marcin Golebiowski, Krzysztof Sroka, Marek Chrusciel
 */
public class EvPopulationGenerating implements
    EvOperator<EvBinaryVectorIndividual> {

  /**
   * Solution space for binary strings
   */
  private EvBinaryVectorSpace space;

  /**
   * Structure for bits partition
   */
  private EvStructureDiscover struct_update;


  /**
   * constructor
   * 
   * @param struct_update structure for bits partition
   * @param space soltion space
   */
  public EvPopulationGenerating(EvStructureDiscover struct_update,
      EvBinaryVectorSpace space) {
    this.struct_update = struct_update;
    this.space = space;
  }


  /**
   * Generate new population using <code>space</code> and
   * <code>struct_update</code>
   */
  @SuppressWarnings("unchecked")
  public EvPopulation<EvBinaryVectorIndividual> apply(
      EvPopulation<EvBinaryVectorIndividual> population) {

    System.out.println("Population generating start");

    EvECGAStructure ecga_struct = struct_update.getStruct();
    EvPopulation<EvBinaryVectorIndividual> new_pop =
        new EvPopulation<EvBinaryVectorIndividual>(population.size());

    /** copy population */
    EvObjectiveFunction<EvBinaryVectorIndividual> function =
        population.get(0).getObjectiveFunction();
    for (int i = 0; i < population.size(); i++) {
      EvBinaryVectorIndividual ind =
          new EvBinaryVectorIndividual(space.getDimension());
      ind.setObjectiveFunction(function);
      new_pop.add(ind);
    }

    /** for each block in ECGAStructure */
    for (int block = 0; block < ecga_struct.getSetCount(); block++) {
      /** get indexes stored in block */
      ArrayList<Integer> positions = ecga_struct.getBlock(block).getElements();

      /** add to list values of all individuals on given indexes */
      ArrayList<int[]> list = new ArrayList<int[]>();
      for (EvBinaryVectorIndividual individual : population) {
        list.add(individual.getBits(positions));
      }

      /** random permutation of list values */
      Collections.shuffle(list);

      /**
       * set shuffled values to individuals in new population at
       * <code>positions</code>
       */
      for (int i = 0; i < list.size(); i++) {
        new_pop.get(i).setBits(positions, list.get(i));
      }
    }

    int count = 0;
    for (int i = 0; i < new_pop.size(); i++) {
      if (new_pop.get(i).isObjectiveFunctionValueCalculated()) {
        count++;
      }
    }

    System.out.println("Obliczony jest" + count);

    System.out.println("Population generating end");
    return new_pop;
  }

}
