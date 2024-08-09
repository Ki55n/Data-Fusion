import Table from "@/components/Table";
import React from "react";

export default function CustomerTable({data}: any) {

   return <Table
       columns={[
           {title: '', key: 'checkbox', width: 'w-[50p]'},
           {title: 'Given Name', key: 'given_name', width: 'w-[50p]'},
           {title: 'Family Name', key: 'family_name', width: 'w-[50p]'},
           {title: 'Email Address', key: 'email', width: 'w-[50p]'},
              {title: 'Birth Date', key: 'birth_date', width: 'w-[50p]'},
                {title: 'Created At', key: 'created_at', width: 'w-[50p]'},
                {title: 'Address', key: 'address', width: 'w-[50p]'},
                {title: 'Locality', key: 'locality', width: 'w-[50p]'},
                {title: 'Postal Code', key: 'postal_code', width: 'w-[50p]'},
       ]}
       data={data || []}
   />
}
