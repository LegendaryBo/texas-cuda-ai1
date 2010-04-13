package wevo.statystyki;

import generator.GeneratorGraczyZGeneracji;

import java.io.IOException;

import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.chart.EvChartTools;
import pl.wroc.uni.ii.evolution.chart.EvGenesAverageValuesChart;
import pl.wroc.uni.ii.evolution.chart.EvObjectiveFunctionValueMaxAvgMinChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.EvBinaryGenesAvgValueGatherer;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;
import Gracze.gracz_v3.GeneratorRegulv3;

/**
 * klasa z narzedziamy do statystyk
 * 
 */
public class StatystykiTools {

	public static void przygotujAlgorytm(EvPersistentSimpleStorage storage,
			EvAlgorithm<EvBinaryVectorIndividual> genericEA) {

		genericEA.addOperatorToEnd(new EvBinaryGenesAvgValueGatherer(
				GeneratorRegulv3.rozmiarGenomu, storage));
		genericEA
				.addOperatorToEnd(new EvObjectiveFunctionValueMaxAvgMinGatherer<EvBinaryVectorIndividual>(
						storage));
		genericEA
				.addOperatorToEnd(new EvStatystykiWygranychIPrzegranychGatherer(
						storage));
		genericEA.addOperatorToEnd(new EvStatystykiLicznikPassowGatherer(
				storage));
		genericEA.addOperatorToEnd(new EvStatystykiKartWygranegoGatherer(
				storage));
		genericEA.addOperatorToEnd(new EvStatystykaWygranychOdKartGatherer(
				storage));
		genericEA
				.addOperatorToEnd(new EvStatystykaWymaganychGlosowR1Gatherer(
						storage));
		genericEA
				.addOperatorToEnd(new EvStatystykaStawkiR1Gatherer(
						storage));
		genericEA.addOperatorToEnd(new EvStatystykaStawkiRXGatherer(
				storage, 2));
		// genericEA.addOperatorToEnd(new
		// EvStatystykaStawkiRXGatherer(storage, 3));
		// genericEA.addOperatorToEnd(new
		// EvStatystykaStawkiRXGatherer(storage, 4));

		genericEA
				.addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
						System.out));

	}

	public static void wypiszStatystyki(EvPersistentStatisticStorage storage,
			int generacja) {

		boolean statystykiGry = true;

		EvStatistic[] stats = EvStatisticFilter
				.byClass(EvGenesAvgValueStatistic.class, storage
						.getStatistics());
		EvStatistic[] stats2 = EvStatisticFilter.byClass(
				EvObjectiveFunctionValueMaxAvgMinStatistic.class,
				storage.getStatistics());
		EvStatistic[] stats3 = null;
		EvStatistic[] stats4 = null;
		EvStatistic[] stats5 = null;
		EvStatistic[] stats6 = null;
		EvStatistic[] stats7 = null;
		EvStatistic[] stats8 = null;
		EvStatistic[] stats9 = null;
		EvStatistic[] stats10 = null;
		EvStatistic[] stats11 = null;
		if (statystykiGry) {

			stats3 = EvStatisticFilter.byClass(
					EvStatystykiWygranychIPrzegranychStatistic.class,
					storage.getStatistics());
			stats4 = EvStatisticFilter.byClass(
					EvStatystykiLicznikPassowStatistic.class, storage
							.getStatistics());
			stats5 = EvStatisticFilter.byClass(
					EvStatystykiKartWygranegoStatistic.class, storage
							.getStatistics());
			stats6 = EvStatisticFilter.byClass(
					EvStatystykaWygranychOdKartStatistic.class,
					storage.getStatistics());
			stats7 = EvStatisticFilter.byClass(
					EvStatystykaWymaganychGlosowR1Statistic.class,
					storage.getStatistics());
			stats8 = EvStatisticFilter.byClass(
					EvStatystykaStawkiR1Statistic.class, storage
							.getStatistics());

			stats9 = EvStatisticFilter.byClass(
					EvStatystykaStawkiRXStatistic.class, storage
							.getStatistics());
			stats10 = EvStatisticFilter.byClass(
					EvStatystykaStawkiRXStatistic.class, storage
							.getStatistics());
			stats11 = EvStatisticFilter.byClass(
					EvStatystykaStawkiRXStatistic.class, storage
							.getStatistics());
		}

		try {

			JFreeChart chart = EvGenesAverageValuesChart
					.createJFreeChart(stats, false,
							GeneratorRegul.indeksyGenowNaWejscie);
			JFreeChart chart2 = EvGenesAverageValuesChart
					.createJFreeChart(stats, false,
							GeneratorRegul.indeksyGenowStawka1);
			JFreeChart chart3 = EvGenesAverageValuesChart
					.createJFreeChart(stats, false,
							GeneratorRegul.indeksyGenowDobijanie1);
			JFreeChart chart4 = EvObjectiveFunctionValueMaxAvgMinChart
					.createJFreeChart(stats2, false);
			if (statystykiGry) {
				JFreeChart chart5 = EvStatystykiWygranychIPrzegranychChart
						.createJFreeChart(stats3, false);
				EvChartTools.createJPG(chart5,
						GeneratorGraczyZGeneracji.SCIEZKA
								+ "generacja_" + generacja
								+ "_wynikiPartii.jpg", 800,
						1300, 100);

				JFreeChart chart6 = EvStatystykiWygranychIPrzegranychBestChart
						.createJFreeChart(stats3, true);
				EvChartTools
						.createJPG(
								chart6,
								GeneratorGraczyZGeneracji.SCIEZKA
										+ "generacja_"
										+ generacja
										+ "_bilansWygranychNajlepszyOs.jpg",
								800, 1300, 100);

				JFreeChart chart11 = EvStatystykiWygranychIPrzegranychBestChart
						.createJFreeChart(stats3, false);
				EvChartTools
						.createJPG(
								chart11,
								GeneratorGraczyZGeneracji.SCIEZKA
										+ "generacja_"
										+ generacja
										+ "_wynikiPartiiNajlepszyOs.jpg",
								800, 1300, 100);

				JFreeChart chart12 = EvStatystykiWygranychIPrzegranychChart
						.createJFreeChart(stats3, true);
				EvChartTools.createJPG(chart12,
						GeneratorGraczyZGeneracji.SCIEZKA
								+ "generacja_" + generacja
								+ "_bilansWygranych.jpg", 800,
						1300, 100);

				JFreeChart chart7 = EvStatystykiLicznikPassowChart
						.createJFreeChart(stats4);
				EvChartTools.createJPG(chart7,
						GeneratorGraczyZGeneracji.SCIEZKA
								+ "generacja_" + generacja
								+ "_statystykiPassow.jpg", 800,
						1300, 100);

				JFreeChart chart8 = EvStatystykiKartWygranegoChart
						.createJFreeChart(stats5);
				EvChartTools
						.createJPG(
								chart8,
								GeneratorGraczyZGeneracji.SCIEZKA
										+ "generacja_"
										+ generacja
										+ "_statystykiKartWygranego.jpg",
								800, 1300, 100);

				JFreeChart chart13 = EvStatystykiKartWygranegoBestChart
						.createJFreeChart(stats5);
				EvChartTools
						.createJPG(
								chart13,
								GeneratorGraczyZGeneracji.SCIEZKA
										+ "generacja_"
										+ generacja
										+ "_statystykiKartWygranegoBest.jpg",
								800, 1300, 100);

				JFreeChart chart9 = EvStatystykaWygranychOdKartChart
						.createJFreeChart(stats6);
				EvChartTools
						.createJPG(
								chart9,
								GeneratorGraczyZGeneracji.SCIEZKA
										+ "generacja_"
										+ generacja
										+ "_statystykiWygranychOdKart.jpg",
								800, 1300, 100);

				JFreeChart chart10 = EvStatystykiLicznikPassowBestChart
						.createJFreeChart(stats4);
				EvChartTools
						.createJPG(
								chart10,
								GeneratorGraczyZGeneracji.SCIEZKA
										+ "generacja_"
										+ generacja
										+ "_statystykiPassowNajlepszyOs.jpg",
								800, 1300, 100);

				JFreeChart chart14 = EvStatystykaWymaganychGlosowR1Chart
						.createJFreeChart(stats7);
				EvChartTools.createJPG(chart14,
						GeneratorGraczyZGeneracji.SCIEZKA
								+ "generacja_" + generacja
								+ "_statystykiWejscieR1.jpg",
						800, 1300, 100);

				JFreeChart chart17 = EvStatystykaWymaganychGlosowR1BestChart
						.createJFreeChart(stats7);
				EvChartTools
						.createJPG(
								chart17,
								GeneratorGraczyZGeneracji.SCIEZKA
										+ "generacja_"
										+ generacja
										+ "_statystykiWejscieR1Best.jpg",
								800, 1300, 100);

				JFreeChart chart15 = EvStatystykaStawkiR1Chart
						.createJFreeChart(stats8);
				EvChartTools.createJPG(chart15,
						GeneratorGraczyZGeneracji.SCIEZKA
								+ "generacja_" + generacja
								+ "_statystykiStawkaR1.jpg",
						800, 1300, 100);

				JFreeChart chart16 = EvStatystykaStawkiR1BestChart
						.createJFreeChart(stats8);
				EvChartTools
						.createJPG(
								chart16,
								GeneratorGraczyZGeneracji.SCIEZKA
										+ "generacja_"
										+ generacja
										+ "_statystykiStawkaR1Best.jpg",
								800, 1300, 100);

				JFreeChart bla = EvStatystykaStawkaRXChart
						.createJFreeChart(stats9);
				EvChartTools.createJPG(bla,
						GeneratorGraczyZGeneracji.SCIEZKA
								+ "generacja_" + generacja
								+ "_statystykiStawkaR2.jpg",
						800, 1300, 100);
				bla = EvStatystykaStawkaRXChart
						.createJFreeChart(stats10);
				EvChartTools.createJPG(bla,
						GeneratorGraczyZGeneracji.SCIEZKA
								+ "generacja_" + generacja
								+ "_statystykiStawkaR3.jpg",
						800, 1300, 100);
				bla = EvStatystykaStawkaRXChart
						.createJFreeChart(stats11);
				EvChartTools.createJPG(bla,
						GeneratorGraczyZGeneracji.SCIEZKA
								+ "generacja_" + generacja
								+ "_statystykiStawkaR4.jpg",
						800, 1300, 100);

				bla = EvStatystykaStawkiRXBestChart
						.createJFreeChart(stats9);
				EvChartTools
						.createJPG(
								bla,
								GeneratorGraczyZGeneracji.SCIEZKA
										+ "generacja_"
										+ generacja
										+ "_statystykiStawkaR2Best.jpg",
								800, 1300, 100);

			}

			EvChartTools.createJPG(chart,
					GeneratorGraczyZGeneracji.SCIEZKA + "generacja_"
							+ generacja
							+ "_indeksy_genow_wejscie1_.jpg",
					800, 1300, 100);
			EvChartTools.createJPG(chart2,
					GeneratorGraczyZGeneracji.SCIEZKA + "generacja_"
							+ generacja
							+ "_indeksy_genow_stawka1_.jpg", 800,
					1300, 100);
			EvChartTools.createJPG(chart3,
					GeneratorGraczyZGeneracji.SCIEZKA + "generacja_"
							+ generacja
							+ "_indeksy_genow_dobijanie1_.jpg",
					800, 1300, 100);
			EvChartTools
					.createJPG(
							chart4,
							GeneratorGraczyZGeneracji.SCIEZKA
									+ "generacja_"
									+ generacja
									+ "_max_avg_dev_min_.jpg",
							800, 1300, 100);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
