package pl.wroc.uni.ii.evolution.tasks;

import pl.wroc.uni.ii.evolution.distribution.strategies.EvIslandModel;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchangeWithServlet;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchanger;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiver;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiverImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvSender;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.ExSenderImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvHyperCube;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvRing;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopology;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvIslandDistribution;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvTaskCreator;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvBlockSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveFunctionDistributionGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticDatabaseSuppportServletStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorCGAOperator;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.EvBinaryGenesAvgValueGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.EvBinaryVectorGenesChangesGatherer;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvBestValueNotImproved;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvLongFunction;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationWithErrorRecovery;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

public class prezentacja implements EvTaskCreator {

  public Runnable create(int task_id, long node_id, String wevo_url) {

    int size=50;
    
    EvAlgorithm<EvBinaryVectorIndividual> algorytm = new 
    EvAlgorithm<EvBinaryVectorIndividual>(100); 
    EvKDeceptiveOneMax objective_function = new EvKDeceptiveOneMax(4);

    algorytm.setObjectiveFunction(objective_function); 
    algorytm.setSolutionSpace(new EvBinaryVectorSpace(objective_function, size)); 
    algorytm.setTerminationCondition(new EvBestValueNotImproved<EvBinaryVectorIndividual>(30)); 
    
    EvBinaryVectorCGAOperator cga_operator = new EvBinaryVectorCGAOperator(50, 0.005, 1000, new EvBinaryVectorSpace(objective_function,size));
    algorytm.addOperator(cga_operator); 


    //we use ring topology
    EvTopology topology = new EvRing(3);  
    EvDBServletCommunicationImpl comm = new EvDBServletCommunicationImpl(wevo_url);
    
    EvPersistentStatisticDatabaseSuppportServletStorage storage = 
      new EvPersistentStatisticDatabaseSuppportServletStorage(task_id, topology.assignCellID() , node_id, comm);    
    algorytm.addOperatorToEnd(new EvBinaryVectorGenesChangesGatherer(storage));
    algorytm.addOperatorToEnd(new EvObjectiveFunctionDistributionGatherer<EvBinaryVectorIndividual>(storage));
    algorytm.addOperatorToEnd(new EvBinaryGenesAvgValueGatherer(size, storage));
    algorytm.addOperatorToEnd(new EvObjectiveFunctionValueMaxAvgMinGatherer<EvBinaryVectorIndividual>(storage));

    
 
    
   
    // THE REST OF EVISLAND STUFF
    
    EvIndividualsExchanger<EvBinaryVectorIndividual> exchanger = new EvIndividualsExchangeWithServlet<EvBinaryVectorIndividual>(new EvDBServletCommunicationWithErrorRecovery(wevo_url, 5, 2000));
    EvReceiver<EvBinaryVectorIndividual> receiver = new EvReceiverImpl<EvBinaryVectorIndividual>(exchanger, topology,  50, 1000, task_id, 15, false); 
    EvSender<EvBinaryVectorIndividual> sender = new ExSenderImpl<EvBinaryVectorIndividual>(exchanger, topology, 50, 1000, task_id, node_id, 10,  false);
    EvIslandModel<EvBinaryVectorIndividual> strategy = new EvIslandModel<EvBinaryVectorIndividual>(receiver, sender); 
    EvIslandDistribution task = new EvIslandDistribution(strategy, algorytm);
 
    return task;
    
    
  }

  
  public static void main(String[] arg) {
    
    EvAlgorithm<EvBinaryVectorIndividual> evolutionary_algorithm = new EvAlgorithm<EvBinaryVectorIndividual>(400);
    
    evolutionary_algorithm.setSolutionSpace(new EvBinaryVectorSpace(new EvOneMax(), 50));
    evolutionary_algorithm.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(100)); 
    
    evolutionary_algorithm.addOperator(new EvBlockSelection<EvBinaryVectorIndividual>(30)); 
    evolutionary_algorithm.addOperator(
        new EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual>());
    evolutionary_algorithm.addOperator(new EvBinaryVectorNegationMutation(0.02));
   
    evolutionary_algorithm.init();
    evolutionary_algorithm.run();
    
    EvBinaryVectorIndividual indiv = evolutionary_algorithm.getBestResult();
    System.out.println(indiv);    
  }
  
  
}


class YourTask2 implements EvTaskCreator {
  public Runnable create(int task_id, long node_id, String wevo_url) {
           EvObjectiveFunction<EvBinaryVectorIndividual> objective_function = new EvOneMax();

    EvAlgorithm<EvBinaryVectorIndividual> genericEA = new EvAlgorithm<EvBinaryVectorIndividual>(3000);
    genericEA.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 1000));
    genericEA.setObjectiveFunction(objective_function);
    genericEA.addOperator(new EvKBestSelection<EvBinaryVectorIndividual>(50));
    genericEA.addOperator(
        new EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual>());
    genericEA.setTerminationCondition(new EvBestValueNotImproved<EvBinaryVectorIndividual>(10));

    EvTopology topology = new EvHyperCube(3);


  EvIndividualsExchanger<EvBinaryVectorIndividual> exchanger = new 
      EvIndividualsExchangeWithServlet<EvBinaryVectorIndividual>(new 
           EvDBServletCommunicationWithErrorRecovery(wevo_url, 5, 2000));


    EvReceiver<EvBinaryVectorIndividual> receiver = new EvReceiverImpl<EvBinaryVectorIndividual>(exchanger, topology,  20, 5000, task_id, 10, false); 
    EvSender<EvBinaryVectorIndividual> sender = new ExSenderImpl<EvBinaryVectorIndividual>(exchanger, topology, 20, 1000, task_id, node_id, 10,  false);


    EvSelection<EvBinaryVectorIndividual> selection = new EvKBestSelection<EvBinaryVectorIndividual>(10); 

    EvReplacement<EvBinaryVectorIndividual> replacement = new EvBestFromUnionReplacement<EvBinaryVectorIndividual>();

    EvIslandModel strategy = new EvIslandModel<EvBinaryVectorIndividual>(receiver, sender, replacement, selection); 
    EvIslandDistribution task = new EvIslandDistribution(strategy, genericEA);

    return task;
     
  }
}


class MasterSlaveTask4 implements EvTaskCreator {
  public Runnable create(int task_id, long node_id,
      String wevo_server_url) {

  int bits = 50;
  int pop_size=100;
  EvTopology topology = new EvRing(3); 
  
  EvAlgorithm<EvBinaryVectorIndividual> alg = new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);
  EvLongFunction<EvBinaryVectorIndividual> one_max = new EvLongFunction<EvBinaryVectorIndividual>(new EvOneMax(), 100);
  EvBinaryVectorSpace solution_space = new EvBinaryVectorSpace(one_max, bits);
  alg.setSolutionSpace(solution_space);
  alg.addOperator(new EvBinaryVectorNegationMutation(0.02));
  alg.addOperator(new EvTournamentSelection<EvBinaryVectorIndividual>(4,2));
  alg.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(8));    


  // THE REST OF EVISLAND STUFF
  EvIndividualsExchanger<EvBinaryVectorIndividual> exchanger = new EvIndividualsExchangeWithServlet<EvBinaryVectorIndividual>(new EvDBServletCommunicationWithErrorRecovery(wevo_server_url, 5, 2000));
  EvReceiver<EvBinaryVectorIndividual> receiver = new EvReceiverImpl<EvBinaryVectorIndividual>(exchanger, topology,  20, 10000, task_id, 10, false); 
  EvSender<EvBinaryVectorIndividual> sender = new ExSenderImpl<EvBinaryVectorIndividual>(exchanger, topology, 20, 10000, task_id, node_id, 10,  false);
  EvIslandModel<EvBinaryVectorIndividual> strategy = new EvIslandModel<EvBinaryVectorIndividual>(receiver, sender); 
  EvIslandDistribution task = new EvIslandDistribution(strategy, alg);
  
  return task;    
  }     
}