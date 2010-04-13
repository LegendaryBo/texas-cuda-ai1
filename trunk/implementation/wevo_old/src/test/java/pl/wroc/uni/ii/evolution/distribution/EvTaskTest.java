package pl.wroc.uni.ii.evolution.distribution;

import org.jmock.Mock;
import org.jmock.cglib.MockObjectTestCase;

import pl.wroc.uni.ii.evolution.distribution.clustering.EvSolutionSpaceLoader;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvIslandDistribution;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.misc.EvIdentity;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import pl.wroc.uni.ii.evolution.testhelper.EvGoalFunction;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;
import pl.wroc.uni.ii.evolution.testhelper.EvStuntSpace;

public class EvTaskTest extends MockObjectTestCase {

	public void testDummy() {
		
	}
	
//  @SuppressWarnings("unchecked")
//  public void testRun() {
//    try {
//
//      EvAlgorithm<EvStunt> alg = new EvAlgorithm<EvStunt>(1);
//      alg.setSolutionSpace(new EvStuntSpace(1));
//      alg.setObjectiveFunction(new EvGoalFunction());
//      alg.setTerminationCondition(new EvMaxIteration<EvStunt>(2));
//      alg.addOperatorToEnd(new EvIdentity<EvStunt>());
//
//      Mock subspace_loader = mock(EvSolutionSpaceLoader.class);
//
//      EvIslandDistribution task = new EvIslandDistribution();
//
//      task.setAlgorithm(alg);
//
//      task
//          .setSolutionSpaceLoader((EvSolutionSpaceLoader<EvBinaryVectorIndividual>) subspace_loader
//              .proxy());
//
//      EvSolutionSpace<EvBinaryVectorIndividual> solution_space = new EvBinaryVectorSpace(
//          new EvOneMax(), 2);
//
//      subspace_loader.expects(atLeastOnce()).method("newSubspaceAvailable")
//          .will(onConsecutiveCalls(returnValue(false), returnValue(true)));
//
//      subspace_loader.expects(atLeastOnce()).method("start").will(
//          returnValue(new Thread()));
//
//      subspace_loader.expects(once()).method("takeSubspace").will(
//          returnValue(solution_space));
//      // algorithm.expects(once()).method("setSolutionSpace").with(eq(solution_space));
//      // algorithm.stubs().method("isTerminationConditionSatisfied").will(
//      // onConsecutiveCalls(returnValue(false),returnValue(false),returnValue(true)));
//      // algorithm.stubs().method("doIteration");
//      // algorithm.expects(exactly(2)).method("init");
//
//      task.run();
//    } catch (Exception ex) {
//      ex.printStackTrace();
//      fail();
//    }
//  }
//
//  @SuppressWarnings("unchecked")
//  public void testRun2() {
//
//    try {
//
//      Mock subspace_loader = mock(EvSolutionSpaceLoader.class);
//
//      EvIslandDistribution task = new EvIslandDistribution();
//
//      EvAlgorithm<EvStunt> alg = new EvAlgorithm<EvStunt>(1);
//      alg.setSolutionSpace(new EvStuntSpace(1));
//      alg.setObjectiveFunction(new EvGoalFunction());
//      alg.setTerminationCondition(new EvMaxIteration<EvStunt>(2));
//      alg.addOperatorToEnd(new EvIdentity<EvStunt>());
//      task.setAlgorithm(alg);
//
//      task
//          .setSolutionSpaceLoader((EvSolutionSpaceLoader<EvBinaryVectorIndividual>) subspace_loader
//              .proxy());
//
//      subspace_loader.expects(atLeastOnce()).method("start").will(
//          returnValue(new Thread()));
//
//      subspace_loader.expects(atLeastOnce()).method("newSubspaceAvailable")
//          .will(returnValue(false));
//      subspace_loader.expects(never()).method("takeSubspace");
//      // algorithm.expects(never()).method("setSolutionSpace").with(eq(solution_space));
//      // algorithm.stubs().method("isTerminationConditionSatisfied").will(
//      // onConsecutiveCalls(returnValue(false),returnValue(false),returnValue(true)));
//      // algorithm.stubs().method("doIteration");
//      // algorithm.expects(once()).method("init");
//
//      task.run();
//
//    } catch (Exception ex) {
//      ex.printStackTrace();
//      fail();
//    }
//
//  }

}
