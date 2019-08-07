
from tariff.bill_calculator import BillCalculator


class Savings:

    def __init__(self, site_id):
        self.bill_calculator = BillCalculator(site_id)


    def get_savings(self, baseline_load, target_load):
        base_bill = self.bill_calculator.calculate_total_bill(baseline_load,
                                                              load_column='building_baseline')
        target_bill = self.bill_calculator.calculate_total_bill(target_load,
                                                                load_column='target')
        return base_bill - target_bill


    def get_demand_savings(self, baseline_load, target_load):
        base_bill = self.bill_calculator.calculate_demand_bill(baseline_load,
                                                               load_column='building_baseline')
        target_bill = self.bill_calculator.calculate_demand_bill(target_load,
                                                                 load_column='target')
        return base_bill - target_bill


    def get_energy_savings(self, baseline_load, target_load):
        base_bill = self.bill_calculator.calculate_energy_bill(baseline_load,
                                                               load_column='building_baseline')
        target_bill = self.bill_calculator.calculate_energy_bill(target_load,
                                                                 load_column='target')
        return base_bill - target_bill