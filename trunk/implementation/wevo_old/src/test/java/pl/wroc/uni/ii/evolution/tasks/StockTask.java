package pl.wroc.uni.ii.evolution.tasks;

import pl.fdm.engine.Quotations;
import pl.fdm.engine.TradingRules;
import pl.powermarket.eg4eem.objectivefunctions.PowerMarketSimpleObjectiveFunction;
import pl.wroc.uni.ii.evolution.distribution.eval.EvExternalEvaluationOperator;
import pl.wroc.uni.ii.evolution.distribution.strategies.EvIslandModel;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchangeWithServlet;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchanger;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiver;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiverImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvSender;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.ExSenderImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvRing;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopology;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvIslandDistribution;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvTaskCreator;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvApplyOnSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveFunctionDistributionGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticDatabaseSuppportServletStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.EvBinaryGenesAvgValueGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.EvBinaryVectorGenesChangesGatherer;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvECGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationWithErrorRecovery;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import external.powermarket.examples.DataTaker;

public class StockTask implements EvTaskCreator {
  
  static final String prefix = "H:\\kozak\\data\\";  
  
  public static void main(String[] args) {
    Quotations quotations = null;
    TradingRules rules = null;    
    
    quotations = DataTaker.takeQuotations(prefix+"quotations\\polpx-2005_apr-may.csv",8);
    rules = DataTaker.takeRules(prefix+"rules\\RULE_SET_1_350.TXT");

    System.out.println("Loaded quotations & rules.");
    EvObjectiveFunction<EvBinaryVectorIndividual> objective_function = new PowerMarketSimpleObjectiveFunction(quotations, rules);
    System.out.println("ObjectiveFunction  created");
    // algorytm
    EvAlgorithm<EvBinaryVectorIndividual> genericEA = new EvECGA(false, 512, 16, 4);
    genericEA.setSolutionSpace(new EvBinaryVectorSpace(objective_function, rules.size()));
    genericEA.setObjectiveFunction(objective_function);
    genericEA.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(30));
    genericEA.addOperatorToBeginning(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(System.out));
    
    //genericEA.
    
    genericEA.init();
    genericEA.run();
    
  }
  
  
  
  
  
  // this warning is removed because it uses contructor designed for test purposes only
  @SuppressWarnings("deprecation")
  public Runnable create(int task_id, long node_id, String wevo_server_url) {

      // komunikator
      EvDBServletCommunicationImpl comm = new EvDBServletCommunicationImpl(wevo_server_url);
      System.out.println("DBServlet communicator created");
      
      //topologia
      EvTopology topology = new EvRing(3);
      System.out.println("Topology  created");
      
      // funkcja celu
//      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function = new EvKDeceptiveOneMax(4);
      Quotations quotations = null;
      TradingRules rules = null;
      
      quotations = DataTaker.takeQuotations(
            prefix+"quotations\\polpx-2005_apr-may.csv",
            8);
      rules = DataTaker.takeRules(prefix+"rules\\RULE_SET_1_350.TXT");
      
      System.out.println("Loaded quotations & rules.");
      
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function = new PowerMarketSimpleObjectiveFunction(quotations, rules);
       
      System.out.println("ObjectiveFunction  created");
     
      // storage
      EvPersistentStatisticDatabaseSuppportServletStorage storage = 
        new EvPersistentStatisticDatabaseSuppportServletStorage(task_id, topology.assignCellID() , node_id, comm);
               
      System.out.println("PersistentStatisticDatabaseStorage  created");

      // external eval
      EvExternalEvaluationOperator<EvBinaryVectorIndividual> eval = 
        new EvExternalEvaluationOperator<EvBinaryVectorIndividual>(1000, task_id, node_id, wevo_server_url);
     
      int n = 2048;
      int d = rules.size();
      
      // algorytm
      EvAlgorithm<EvBinaryVectorIndividual> genericEA = new EvAlgorithm<EvBinaryVectorIndividual>(n);
      genericEA.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(30));
      genericEA.setSolutionSpace(new EvBinaryVectorSpace(objective_function, d));
      
      genericEA.addOperatorToEnd(eval);
      genericEA.addOperatorToEnd(new EvReplacementComposition<EvBinaryVectorIndividual>(
         new EvApplyOnSelectionComposition<EvBinaryVectorIndividual>
         (
             new EvTournamentSelection<EvBinaryVectorIndividual>(4, 1),
             new EvTwoOperatorsComposition<EvBinaryVectorIndividual>
             (eval, 
                 new EvTwoOperatorsComposition<EvBinaryVectorIndividual>(
                    new EvBinaryVectorNegationMutation(0.01), 
                    new EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual>()))),
         new EvBestFromUnionReplacement<EvBinaryVectorIndividual>()));
      genericEA.addOperatorToEnd(eval);
      
      genericEA.addOperatorToEnd(new EvBinaryVectorGenesChangesGatherer(storage));
      genericEA.addOperatorToEnd(new EvObjectiveFunctionDistributionGatherer<EvBinaryVectorIndividual>(storage));
      genericEA.addOperatorToEnd(new EvBinaryGenesAvgValueGatherer(rules.size(), storage));
      genericEA.addOperatorToEnd(new EvObjectiveFunctionValueMaxAvgMinGatherer<EvBinaryVectorIndividual>(storage));
      genericEA.addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(System.out));
      
      System.out.println("Algorithm created");
      
      // exchanger
      EvIndividualsExchanger<EvBinaryVectorIndividual> exchanger = new EvIndividualsExchangeWithServlet<EvBinaryVectorIndividual>(new EvDBServletCommunicationWithErrorRecovery(comm, 5, 2000));
      System.out.println("Exchanger created");

      // strategy
      EvReceiver<EvBinaryVectorIndividual> receiver = new EvReceiverImpl<EvBinaryVectorIndividual>(exchanger, topology,  10, 50000, task_id, 10, false); 
      EvSender<EvBinaryVectorIndividual> sender = new ExSenderImpl<EvBinaryVectorIndividual>(exchanger, topology, 10, 30000, task_id, node_id, 10,  false);

      
      EvSelection<EvBinaryVectorIndividual> selection = new EvKBestSelection<EvBinaryVectorIndividual>(10); 
      EvReplacement<EvBinaryVectorIndividual> replacement = new EvBestFromUnionReplacement<EvBinaryVectorIndividual>();
      EvIslandModel<EvBinaryVectorIndividual> strategy = new EvIslandModel<EvBinaryVectorIndividual>(receiver, sender, replacement, selection); 
      
      System.out.println("Strategy created");
      
      // tworzenie EvIsland
      EvIslandDistribution task = new EvIslandDistribution();
      task.setAlgorithm(genericEA);
      task.setDistributedStrategy(strategy);
     
      System.out.println("EvIsland created");
      return task;
    }
 }