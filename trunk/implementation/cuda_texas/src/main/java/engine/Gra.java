package engine;

import engine.rezultaty.Rezultat;
import generator.GeneratorRozdan;

import java.util.ArrayList;

import Gracze.Gracz;
import Gracze.GraczCudaTest;
import Gracze.GraczStalaStawka;

/**
 * 
 * Klasa reprezentujaca gre w texas holdem.
 * 
 * Obiekt nadaje sie do wielokrotnego przeprowadzania gier (z tymi samymi
 * przeciwnikami)
 * 
 * @author Kacper Gorskir (railman85@gmail.com)
 * 
 */
public class Gra {

	// tablica 6 graczy
	public Gracz[] gracze;
	// czy wlaczone jest logowanie
	private boolean raporting = false;

	// poczatkowa wartosc minimalnego bida
	public double minimal_bid = 10;
	// bidy kazdego z graczy
	public double[] bids = null;
	// true - jesli dany gracz spasowal
	public boolean[] pass = null;
	// akutalna stawka (to nie to samo co pula)
	public double stawka = minimal_bid;

	// czyja kolej na wylozenie musu
	// mus to takie przymusowe wejscie w zerowej rundzie
	public int kto_na_musie = 4;

	// zmienna mowi, ktora jest aktualnie runda
	// 0 - dwie karty w reku
	// 1 - dwie karty i 3 publiczne
	// 2 - dwie karty i 4 publiczne
	// 3 - dwie karty i 5 publicznych
	public int runda = 0;
	// ile pieniedzy jest aktualnie w puli
	public double pula = 0.0d;

	// aktualna ilosc graczy w grze (na poczatku 6, potem sie wykruszaja)
	public int graczy_w_grze = 6;

	// to jest obiekt odpowiedzialny za rozdania
	public GeneratorRozdan rozdanie = null;

	public int[] pass_runda = new int[6];
	public int[] kartaWygranego = new int[6];

	// obiekt z graczami, ktorzy zawsze stawiaja 20.0;
	public Gra(GeneratorRozdan generator, int mode) {
		gracze = new Gracz[6];
		for (int i=0; i < 6; i++) {
			
			if (mode==0)
				gracze[i] = new GraczStalaStawka();
			if (mode==1)
				gracze[i] = new GraczCudaTest(i);
			
			gracze[i].gra=this;
		}
		rozdanie = generator;
	}
	
	public Gra(GeneratorRozdan generator) {
		rozdanie = generator;
		
		bids = new double[6];
		pass = new boolean[6];
		
		bids[kto_na_musie % 6] = minimal_bid;
		bids[(kto_na_musie + 1) % 6] = minimal_bid/2;
	}
	
	/**
	 * Tworzy gre z losowym seedem.
	 * 
	 * @param gracze_
	 *                - tablica DOKLADNIE szesciu graczy
	 */
	public Gra(Gracz[] gracze_) {
		gracze = gracze_;
		if (gracze_!=null)
		for (int i = 0; i < 6; i++) {
			gracze[i].gra = this;
		}
		rozdanie = new GeneratorRozdan();
	}

	/**
	 * Tworzy gre z podanych seedem
	 * 
	 * @param gracze_
	 * @param seed
	 */
	public Gra(Gracz[] gracze_, int seed) {
		gracze = gracze_;
		for (int i = 0; i < 6; i++) {
			gracze[i].gra = this;
		}
		rozdanie = new GeneratorRozdan(seed);
	}

	String log = null;

	/**
	 * Funkcja rozpoczyna JEDNA gre
	 */
	final public String play_round(boolean logowanie) {

		raporting = logowanie;
		if (raporting)
			log = new String();

		/* inicjalizacja zmiennych */
		if (raporting) {
			log += "\n karty: " + rozdanie.toString() + " \n";
			log += " runda 1\n";
		}

		bids = new double[6];
		pass = new boolean[6];
		graczy_w_grze = 6;

		int kto_podbil = -1;

		gracze[kto_na_musie % 6].musik(minimal_bid);
		bids[kto_na_musie % 6] = minimal_bid;

		gracze[(kto_na_musie + 1) % 6].musik(minimal_bid / 2);
		bids[(kto_na_musie + 1) % 6] = minimal_bid/2;

		pula = minimal_bid + minimal_bid / 2;
		stawka = minimal_bid;


		// zaczynami pierwsza runde podbijania
		while (true) {

			for (int i = 0; i < 6; i++) {

				// gdy wszyscy oprocz jednej osoby spasowali
				if (graczy_w_grze == 1) {
					wszyscy_spasowali();
					return log;
				}
				
				if (kto_podbil == i) {
					play_round(1);
					return log;
				}

				if (kto_podbil == - 1) {
					kto_podbil = 0;
				}

				if (pass[i] == true)
					continue;

				// gdy wszyscy oprocz jednej osoby spasowali
				if (graczy_w_grze == 1) {
					wszyscy_spasowali();
					return log;
				}

				double bid = gracze[i].play(1, bids[i]);
				// System.out.println(bid);
				if (bid == -1) {
					if (raporting)
						log += "Gracz " + (i + 1)
								+ " spasowal\n";
					pass[i] = true;
					pass_runda[i] = 0;
					graczy_w_grze--;
				} else {
					// if (bid < stawka)
					// throw new
					// IllegalStateException("nieprawidlowy ruch! Stawka "+stawka+" bid:"+bid+" gracz"+i);
					if (bid > stawka) {
						kto_podbil = i;
						stawka = bid;
					}
					if (bid >= stawka) {
						pula += stawka - bids[i];
						bids[i] = bid;
						if (raporting)
							log += "Gracz "
									+ (i + 1)
									+ "  stawia "
									+ bid
									+ " "
									+ pula
									+ "\n";
					}
				}
			}

		}

	}

	// metoda sluzy do informowania jedynego gracza pozostajacego w grze, ze
	// wygral stawke
	private void wszyscy_spasowali() {
		for (int i = 0; i < 6; i++) {
			if (!pass[i])
				gracze[i].wygrana(pula);
		}
	}

	// 0 - gracza maja 2 karty w reku
	// 1 - gracze widza 3 publicnze karty
	// 2 - gracze widze 4 karty
	// 3 - gracze widza 5 kart
	// 4 - koniec gry
	final private void play_round(int round) {

		if (raporting)
			log += "\n runda " + (round + 1) + "\n";

		runda++;
		int kto_podbil = 0;
		for (int i = 5; i > 0; i--)
			if (!pass[i]) {
				kto_podbil = i;
				break;
			}
		stawka = 0;

		if (round == 4) {
			sprawdzanie_kart();
			return;
		}

		bids = new double[6];

//		System.out.println("kto podbil "+kto_podbil);
		int i=kto_podbil;
		boolean pierwszy_gracz=false;
		while (true) {

				
//				System.out.println("sprawdzam gracza nr "+(i%6+1)+" w rundzie "+round);
//				System.out.flush();
				if (kto_podbil == i%6 && pierwszy_gracz) {
					play_round(round + 1);
					return;
				}
				pierwszy_gracz=true;

				if (graczy_w_grze==1) {
					wszyscy_spasowali();
					return;
				}
				
				if (pass[i%6] == true) {
					i++;
					continue;
				}

				double bid = gracze[i%6].play(runda + 1, bids[i%6]);
				if (bid == -1) {
					pass[i%6] = true;
					pass_runda[i%6] = runda;
					graczy_w_grze--;
					if (raporting)
						log += "gracz " + (i%6 + 1)
								+ " spasowal\n";
				} else {
					if (bid < stawka)
						throw new IllegalStateException(
								"nieprawidlowy ruch! Stawka "
										+ stawka
										+ " bid:"
										+ bid
										+ " gracz"
										+ i%6
										+ "\n");
					if (bid > stawka) {
						kto_podbil = i%6;
						stawka = bid;
					}
					if (bid >= stawka) {
						pula += stawka - bids[i%6];
						bids[i%6] = bid;
						if (raporting)
							log += "Gracz "
									+ (i%6 + 1)
									+ " ("
									+ Rezultat
											.pobierzPrognoze(
													this,
													i%6)
									+ ") stawia"
									+ bid
									+ " pula "
									+ pula
									+ " podbil:"
									+ (kto_podbil + 1)
									+ "\n";
					}
				}

				
				i++;
			}

		

	}

	public static int[] sprawdzenie_kart(boolean pass[], GeneratorRozdan rozdanie) {
		
		Rezultat[] rozdania = new Rezultat[6];

		// pobieramy rozdania dla wszystkich graczy aktualnie w grze
		for (int i = 0; i < 6; i++) {
			if (!pass[i]) {
				rozdania[i] = RegulyGry
						.najlepsza_karta(rozdanie
								.getAllCards(i));
			}
		}

		// lista wygranych graczy
		ArrayList<Integer> wygrany = new ArrayList<Integer>();

		// szukamy graczow, ktorzy wygrali partie (moga byc remisy!)
		for (int i = 0; i < 6; i++) {
			if (!pass[i]) {
				if (wygrany.size() == 0)
					wygrany.add(i);
				else {
					int wynik = rozdania[i]
							.porownaj(rozdania[wygrany
									.get(0)]);
					if (wynik == 1) {
			
						wygrany.clear();
						wygrany.add(i);
					}
					if (wynik == 0) {
						wygrany.add(i);
					}
				}
			}
		}	
	
		Integer[] integery = wygrany.toArray(new Integer[0]);
		int[] wygrani = new int[integery.length];
		for (int i=0; i < integery.length; i++)
			wygrani[i] = integery[i];
		
		return wygrani;
	}
	
	/*
	 * sprawdzenie kart
	 */
	private void sprawdzanie_kart() {
		Rezultat[] rozdania = new Rezultat[6];

		// pobieramy rozdania dla wszystkich graczy aktualnie w grze
		for (int i = 0; i < 6; i++) {
			if (!pass[i]) {
				rozdania[i] = RegulyGry
						.najlepsza_karta(rozdanie
								.getAllCards(i));
			}
		}

		// lista wygranych graczy
		ArrayList<Integer> wygrany = new ArrayList<Integer>();

		// szukamy graczow, ktorzy wygrali partie (moga byc remisy!)
		for (int i = 0; i < 6; i++) {
			if (!pass[i]) {
				if (wygrany.size() == 0)
					wygrany.add(i);
				else {
					int wynik = rozdania[i]
							.porownaj(rozdania[wygrany
									.get(0)]);
					if (wynik == 1) {
						for (Integer pierwszy : wygrany) {
							pass[pierwszy] = true;
						}
						wygrany.clear();
						wygrany.add(i);
					}
					if (wynik == 0) {
						wygrany.add(i);
					}
				}
			}
		}
		
		// dzielimy pule pomiedzy graczami
		if (raporting)
			log += "wygrani: \n";
		for (Integer pierwszy : wygrany) {

			if (raporting)
				log += "" + (pierwszy + 1) + "\n";

			gracze[pierwszy].wygrana(pula / wygrany.size());
			kartaWygranego[pierwszy] = rozdania[pierwszy].poziom;
		}

	}

//	private boolean koniec_licytacji() {
//
//		double max_bid = 0;
//		for (int i = 0; i < 6; i++) {
//			if (max_bid < bids[i])
//				max_bid = bids[i];
//		}
//
//		for (int i = 0; i < 6; i++) {
//			if (max_bid > bids[i] && !pass[i])
//				return false;
//		}
//
//		return true;
//	}

	public double getMyBid(int index) {
		return bids[index];
	}

	// zwraca podany numer karte z publicznego stosu
	public Karta getPublicCard(int index) {
//		if (runda == 0 || (runda == 1 && index > 2)
//				|| (runda == 2 && index > 3)
//				|| (runda == 3 && index > 4))
//			throw new IllegalStateException(
//					"brak dostepu do karty " + runda + " "
//							+ index);
		return rozdanie.publiczny_stos[index];
	}

	// zwraca privatna karte podanego gracza
	public Karta getPrivateCard(int kolejnosc, int karta) {
		if (kolejnosc == 0)
			return rozdanie.gracz1[karta];
		if (kolejnosc == 1)
			return rozdanie.gracz2[karta];
		if (kolejnosc == 2)
			return rozdanie.gracz3[karta];
		if (kolejnosc == 3)
			return rozdanie.gracz4[karta];
		if (kolejnosc == 4)
			return rozdanie.gracz5[karta];
		if (kolejnosc == 5)
			return rozdanie.gracz6[karta];
		return null;
	}

}
