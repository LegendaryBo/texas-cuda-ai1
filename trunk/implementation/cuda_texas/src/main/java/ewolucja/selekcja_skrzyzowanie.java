package ewolucja;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorMultiPointCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import wevo.TaxasSolutionSpace;
import wevo.TexasIndividual;
import wevo.TexasObjectiveFunction;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

/**
 * 
 * Program main robiacy ewolucje polegajaca na krzyzowaniu
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 *
 */
public class selekcja_skrzyzowanie {

  public static void main(String[] args) {
    
    GeneratorRegul.init();
    TexasObjectiveFunction objective_function = new TexasObjectiveFunction(2000, 10, true, true);
    int populacja = 200;
    int genes = GeneratorRegul.rozmiarGenomu;
    int iteracji = 100;
    
    EvAlgorithm<EvBinaryVectorIndividual> genericEA =
        new EvAlgorithm<EvBinaryVectorIndividual>(populacja);
    
    genericEA.setSolutionSpace( new TaxasSolutionSpace(objective_function, 3, 10) );
    genericEA.setObjectiveFunction(objective_function);
  
    EvBestFromUnionReplacement replacement = new EvBestFromUnionReplacement();
    //EvEliteReplacement replacement = new EvEliteReplacement<EvBinaryVectorIndividual>(populacja);
    EvKnaryVectorMultiPointCrossover cross = new EvKnaryVectorMultiPointCrossover(2);
    EvBinaryVectorNegationMutation mut = new EvBinaryVectorNegationMutation(0.01);
    
    genericEA.addOperator(new EvReplacementComposition(
        new EvTwoOperatorsComposition(cross, mut)
        , replacement));
    
    
  
    genericEA.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(
          iteracji));    
  
  
    genericEA.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
          System.out));  
  
  
    genericEA.init();
    genericEA.run();
    System.out.println(TexasIndividual.describe(genericEA.getBestResult()));
    
    
    
    
  }
  
  
}
