package pl.wroc.uni.ii.evolution.engine.samplealgorithms;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga.EvPopulationGenerating;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga.EvStructureDiscover;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * @see pl.wroc.uni.ii.evolution.engine.operators.spacespecific.binaryvector.ECGAOperator
 * @author Marcin Golebiowski, Krzysztof Sroka, Marek Chrusciel
 */
public class EvECGA extends EvAlgorithm<EvBinaryVectorIndividual> {

  private boolean use_previous_structure;

  private EvOperator<EvBinaryVectorIndividual> selection;


  /**
   * Constructs a ECGA algorithm with EvTournament selection.
   * 
   * @param use_previous_structure
   *        <ul>
   *        <li> if <code> true </code> then discovery of the best structure
   *        uses structure from previous iteration
   *        <li> if <code> false </code> then discovery of the best structure is
   *        done from initial structure
   *        </ul>
   * @param population_size
   * @param tournament_size
   * @param number_of_winners
   */
  public EvECGA(boolean use_previous_structure, int population_size,
      int tournament_size, int number_of_winners) {
    super(population_size);
    this.use_previous_structure = use_previous_structure;
    this.selection =
        new EvTournamentSelection<EvBinaryVectorIndividual>(tournament_size,
            number_of_winners);
  }


  @Override
  public void init() {
    super.init();
    EvStructureDiscover struct_update =
        new EvStructureDiscover((EvBinaryVectorSpace) solution_space,
            use_previous_structure);
    EvPopulationGenerating pop_gen =
        new EvPopulationGenerating(struct_update,
            (EvBinaryVectorSpace) solution_space);

    super.addOperatorToEnd(selection);
    super.addOperatorToEnd(struct_update);
    super.addOperatorToEnd(pop_gen);
  }
}
