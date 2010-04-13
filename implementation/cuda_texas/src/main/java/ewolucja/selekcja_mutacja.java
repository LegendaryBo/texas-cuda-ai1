package ewolucja;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import wevo.TaxasSolutionSpace;
import wevo.TexasObjectiveFunction;
import Gracze.GraczAIv2;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

/**
 * 
 * Program main robiacy ewolucje uzywajacej mutacji jako operatora ewolucyjnego
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 *
 */
public class selekcja_mutacja {

  public static void main(String[] args) {
    
    GeneratorRegul.init();
    
    TexasObjectiveFunction objective_function = new TexasObjectiveFunction(2000, 9, true, true);
    
    
    final int populacja = 100;
    final int eliteSize = 30;
    final int genes = GeneratorRegul.rozmiarGenomu;
    final int iteracji = 300;

    
    
    EvAlgorithm<EvBinaryVectorIndividual> genericEA =
        new EvAlgorithm<EvBinaryVectorIndividual>(populacja);
    
    genericEA.setSolutionSpace(
        new TaxasSolutionSpace(objective_function, 3, 10));
    genericEA.setObjectiveFunction(objective_function);
  
    EvBestFromUnionReplacement replacement = new EvBestFromUnionReplacement<EvBinaryVectorIndividual>();
    //EvKnaryVectorUniformCrossover cross = new EvKnaryVectorUniformCrossover();
    EvBinaryVectorNegationMutation mutation = new EvBinaryVectorNegationMutation(0.01);
    mutation.setMutateClone(true);
    genericEA.addOperator(new EvReplacementComposition(mutation, replacement));

  
    genericEA.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(
          iteracji));    
  
  
    genericEA.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
          System.out));  
  
  
    genericEA.init();
    genericEA.run();
    
    
    EvBinaryVectorIndividual indi = genericEA.getBestResult();
    GraczAIv2 ai = new GraczAIv2(indi, 0);
    System.out.println(ai);
//    String ret = new String();
//    for (int i=0; i < 48; i++)
//      ret +=indi.getGene(i);
//    System.out.println(ret); 
//    ret = "";
//    for (int i=48; i < 107; i++)
//      ret +=indi.getGene(i);
//    System.out.println(ret);
    
    
//    TexasObjectiveFunction function2 = new TexasObjectiveFunction(5000, 0);
//    TexasObjectiveFunction function3 = new TexasObjectiveFunction(5000, 1);
//    
//    System.out.println(objective_function.getStats(genericEA.getBestResult()));
//    System.out.println("");
//    
//    EvBinaryVectorIndividual ind = genericEA.getBestResult();
//    ind.setObjectiveFunction(function2);
//    
//    System.out.println("trudnosc mala: "+ind.getObjectiveFunctionValue());
//    System.out.println(function2.getStats(ind));    
//    
//    ind.setObjectiveFunction(function3);
//    
//    System.out.println("trudnosc srednia: "+ind.getObjectiveFunctionValue());
//    System.out.println(function3.getStats(ind));     
      
    
    
  }
  
}
