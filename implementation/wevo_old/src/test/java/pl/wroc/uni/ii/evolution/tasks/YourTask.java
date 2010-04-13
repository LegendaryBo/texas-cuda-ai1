package pl.wroc.uni.ii.evolution.tasks;
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
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvECGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvBestValueNotImproved;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvLongFunction;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationWithErrorRecovery;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;


public class YourTask implements EvTaskCreator {
    public Runnable create(int task_id, long node_id,
        String wevo_server_url) {
      
      //topologia
      EvTopology topology = new EvRing(3);
      System.out.println("Topology  created");
     

      // funkcja celu
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function = 
        new EvLongFunction<EvBinaryVectorIndividual>(new  EvKDeceptiveOneMax(4), 100); 
       
      System.out.println("ObjectiveFunction  created");
      
      // external eval
      EvExternalEvaluationOperator<EvBinaryVectorIndividual> eval = 
        new EvExternalEvaluationOperator<EvBinaryVectorIndividual>(2000, task_id, node_id, wevo_server_url);
     
      
      // storage
      //EvPersistentStatisticDatabaseSuppportServletStorage storage = 
      //  new EvPersistentStatisticDatabaseSuppportServletStorage(task_id, topology.assignCellID() , node_id, wevo_server_url);
               
      System.out.println("PersistentStatisticDatabaseStorage  created");

      
      // algorytm
      EvAlgorithm<EvBinaryVectorIndividual> genericEA = new EvECGA(false, 100, 16, 4);
      genericEA.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 120));
      genericEA.setObjectiveFunction(objective_function);
      genericEA.setTerminationCondition(new EvBestValueNotImproved<EvBinaryVectorIndividual>(5));
      
      genericEA.setFirstOperator(eval);
      genericEA.setLastOperator(eval);
      
      /*genericEA.addOperator(new EvBinaryVectorGenesChangesGatherer(storage));
      genericEA.addOperator(new EvObjectiveFunctionDistributionGatherer<EvBinaryVectorIndividual>(storage));
      genericEA.addOperator(new EvBinaryGenesAvgValueGatherer(60, storage));
      genericEA.addOperator(new EvObjectiveFunctionValueMaxAvgMinGatherer<EvBinaryVectorIndividual>(storage));

      genericEA.addOperator(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(System.out));*/
      System.out.println("Algorithm created");
      
      // exchanger
      EvIndividualsExchanger<EvBinaryVectorIndividual> exchanger = new EvIndividualsExchangeWithServlet<EvBinaryVectorIndividual>(new EvDBServletCommunicationWithErrorRecovery(wevo_server_url, 5, 2000));

      System.out.println("Exchanger created");

      // strategy
      EvReceiver<EvBinaryVectorIndividual> receiver = new EvReceiverImpl<EvBinaryVectorIndividual>(exchanger, topology,  20, 10000, task_id, 10, false); 
      EvSender<EvBinaryVectorIndividual> sender = new ExSenderImpl<EvBinaryVectorIndividual>(exchanger, topology, 20, 10000, task_id, node_id, 10,  false);

      
      EvSelection<EvBinaryVectorIndividual> selection = new EvKBestSelection<EvBinaryVectorIndividual>(10); 
      EvReplacement<EvBinaryVectorIndividual> replacement = new EvBestFromUnionReplacement<EvBinaryVectorIndividual>();
      EvIslandModel<EvBinaryVectorIndividual> strategy = new EvIslandModel<EvBinaryVectorIndividual>(receiver, sender, replacement, selection); 
      
      System.out.println("Strategy created");
      
      // tworzenie EvIsland
      EvIslandDistribution task = new EvIslandDistribution();
      task.setAlgorithm(genericEA);
      task.setDistributedStrategy(strategy);
      task.setExternalEval(eval);
     
      System.out.println("EvIsland created");
      return task;
    }
 }