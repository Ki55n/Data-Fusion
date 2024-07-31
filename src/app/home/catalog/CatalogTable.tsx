import Table from "@/components/Table";
import React from "react";
//import suspense
import {Suspense} from "react";
import axios from "axios";
import {getCatalog, GenericResponse, Product} from "@/services/catalog";
import Loader from "@/components/loader";
import UpdateDialog from "@/app/home/catalog/updateDialog";

export default async function CatalogTable({data}: any) {
  
    // try{
    //     const catalog: GenericResponse<Product>  = await getCatalog()
    //     const data = catalog.data.map((item) => {
    //         return {
    //             checkbox: <input type="checkbox"/>,
    //             ...item,
    //             tools: <UpdateDialog product={item}/>
    //         }
    //     })
    //     console.log(catalog)
    // }catch (e) {
    //     console.log(e)
    //     return <div>error</div>
    // }
    
    // const data: any[] = []; // Declare the 'data' variable as an empty array of type 'any[]'
    return <Table
       columns={[
           {title: '', key: 'checkbox', width: 'w-[50p]'},
           {title: 'Name', key: 'name', width: 'w-[50p]'},
           {title: 'Price', key: 'price', width: 'w-[25p]'},
           {title: 'Description', key: 'description', width: 'w-[25p]'},
           {title: '', key: 'tools', width: 'w-[25p]'},
       ]}
       data={data || []}
   />
}
