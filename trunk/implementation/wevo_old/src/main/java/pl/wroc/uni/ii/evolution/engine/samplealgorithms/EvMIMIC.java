package pl.wroc.uni.ii.evolution.engine.samplealgorithms;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorMIMICOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;

/**
 * MIMIC algorithm with KB_SELECTION
 * 
 * @author Marek Chrusciel, Michal Humenczuk
 */
public class EvMIMIC extends EvAlgorithm<EvBinaryVectorIndividual> {

  private EvSelection<EvBinaryVectorIndividual> selection;


  /**
   * @param population_size
   */
  public EvMIMIC(int population_size,
      EvSelection<EvBinaryVectorIndividual> selection) {
    super(population_size);
    this.selection = selection;
  }


  @Override
  public void init() {

    /** Create new MIMIC operator */
    EvBinaryVectorMIMICOperator mop =
        new EvBinaryVectorMIMICOperator(population_size);

    super.addOperatorToEnd(selection);
    super.addOperatorToEnd(mop);

    super.init();
  }
}
